#define CL_EPS 0x1.0p-23f

#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#include "argtable3.h"

#include "Buttw.hpp"
#include "PhaseDiffAndAmp.hpp"
#include "AmpPhaseShift.hpp"

using namespace std;

void printDims(cv::UMat in) {
    std::string r;
    uchar depth = in.type() & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (in.type() >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
    cout << r << ":" << in.rows << "x" << in.cols << "x" << in.channels() << endl;
}

int main(int argc, char** argv) {
    // TODO: args for sigma, no-vis, grey vid, outfile, bp-lo, bp-hi, device, pyr_levels
    struct arg_lit* help = arg_lit0(NULL, "help", "print this help and exit");
    struct arg_lit* version = arg_lit0(NULL, "version", "print version information and exit");
    struct arg_lit* novis = arg_lit0(NULL, "no-vis", "don't visualise the output");
    struct arg_file* infile = arg_filen("i", NULL, "input", 1, 1, "input file");
    struct arg_file* outfile = arg_filen("o", NULL, "output", 0, 1, "output file");
    struct arg_int* sigma = arg_int0(NULL, "sigma", "<odd n>", "sigma used for spatial blurring of filtred phase (default 3)");
    struct arg_int* alpha = arg_int0(NULL, "alpha", "<n>", "amplification factor (default 200)");
    struct arg_int* N = arg_int0(NULL, "N", "<n>", "order of filter (default 3)");
    struct arg_dbl* flo = arg_dbln(NULL, "flo", "<double>", 1, 1, "lower frequency of badpass filter (Hz)");
    struct arg_dbl* fup = arg_dbln(NULL, "fup", "<double>", 1, 1, "opper frequency of badpass filter (Hz)");
    struct arg_int* fs = arg_int0(NULL, "fs", "<n>", "overwrite sample rate derived from video info");
    struct arg_end* end = arg_end(20);
    void* argtable[] = { help,version,novis,infile,outfile,sigma,alpha,N,flo,fup,fs,end };

    // Set default values
    sigma->ival[0] = 3;
    N->ival[0] = 3;
    alpha->ival[0] = 200;

    const char* arg_progname = "omm";
    int arg_nerrors;
    int arg_exitcode = 0;

    /* Parse the command line as defined by argtable[] */
    arg_nerrors = arg_parse(argc, argv, argtable);

    /* special case: '--help' takes precedence over error reporting */
    if (help->count > 0) {
        printf("Usage: %s", arg_progname);
        arg_print_syntax(stdout, argtable, "\n");
        printf("Demonstrate command-line parsing in argtable3.\n\n");
        arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        exit(0);
    }
    
    /* If the parser returned any errors then display them and exit */
    if (arg_nerrors > 0) {
        /* Display the error details contained in the arg_end struct.*/
        arg_print_errors(stdout, end, arg_progname);
        printf("Try '%s --help' for more information.\n", arg_progname);
        arg_exitcode = 1;
        //goto exit;
    }

    if (!cv::ocl::haveOpenCL()) {
        cout << "OpenCL is not avaiable..." << endl;
        return -1;
    }
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        cout << "Failed creating the context..." << endl;
        return -1;
    }

    cout << context.ndevices() << " GPU devices are detected." << endl;
    for (int i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);
        cout << "name                 : " << device.name() << endl;
        cout << "available            : " << device.available() << endl;
        cout << "imageSupport         : " << device.imageSupport() << endl;
        cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
        cout << endl;
    }

    // Select the first device
    cv::ocl::Device(context.device(0));

    const unsigned int pyr_levels = 4;
    cv::UMat frame_raw, frame;    
    std::vector<cv::UMat> frame_channels;

    // Variables dealing the Riesz pyramid
    std::vector<cv::UMat> pyramid;
    std::vector<cv::UMat*> laplacian(pyr_levels), laplacian_prev(pyr_levels), laplacian_amp(pyr_levels);
    std::vector<cv::UMat*> riesz(pyr_levels), riesz_prev(pyr_levels);
    std::vector<cv::UMat> pyr_ups(pyr_levels);

    std::vector<cv::UMat> state(pyr_levels); // Store the state of the temporal filter
    std::vector <cv::UMat> phase(pyr_levels), amp(pyr_levels), phase_diff(pyr_levels), phase_filt(pyr_levels);

    // Variables used for spatial blurring of the filtred phase
    std::vector<std::vector<cv::UMat>> riesz_chans(pyr_levels, std::vector<cv::UMat>(2));
    std::vector<cv::UMat> denominator(pyr_levels), numerator(pyr_levels), denominator2ch(pyr_levels), amp2ch(pyr_levels);

    unsigned int frame_counter = 0;

    // Riesz approximations, basically an edge detector
    cv::Mat riesz_kernel_x(1, 3, CV_32F);
    riesz_kernel_x.at<float>(0, 0) = 0.5;
    riesz_kernel_x.at<float>(0, 1) = 0.0;
    riesz_kernel_x.at<float>(0, 2) = -0.5;
    cv::Mat riesz_kernel_y = riesz_kernel_x.t();
    
    cv::VideoCapture cap(infile->filename[0]);
    if (!cap.isOpened()) {
        cout << "Failed to open video file." << endl;
        exit(1);
    }

    // Set sample rate if it is not specified on command-line
    if (fs->count == 0) {
        fs->ival[0] = cap.get(cv::CAP_PROP_FPS);
        cout << "Video fps: " << fs->ival[0] << endl;
    }

    cv::VideoWriter vidout;

    if (!novis->count) {
        cv::namedWindow("w", 1);
    }

    cv::Buttw butt(N->ival[0], fs->ival[0], flo->dval[0], fup->dval[0]);
    cv::PhaseDiffAndAmp pdaa;
    cv::AmpPhaseShift aps(alpha->ival[0]);

    bool first_run = true;
    std::chrono::steady_clock::time_point t_start =std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_end;

    while (1) {
        cap >> frame_raw;
        if (frame_raw.empty()) {
            break;
        }
      
        // Measure throughput (fps)
        if (frame_counter % 100 == 0 && !first_run) {
            t_end = std::chrono::steady_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
            std::cout << "FPS: " << 100.0*1e6 / dur << std::endl;
            t_start = std::chrono::steady_clock::now();
        }
        frame_counter++;
        
        // TODO: if colour frame, convert to grey
        cv::split(frame_raw, frame_channels);
        frame_channels[0].convertTo(frame, CV_32FC1, 1.0f/255.0f);

        // Gaussian pyramid will have pyr_levels+1 images, pyramid[0]==frame, 
        cv::buildPyramid(frame, pyramid, pyr_levels);

        // At first run, initialise variables to the correct sizes
        if (first_run) {
            for (int i = 0; i < pyr_levels; i++) {
                cv::Size size = (pyramid[i]).size();
                amp[i].create(size, CV_32FC1);
                laplacian_amp[i] = new cv::UMat(size, CV_32FC1);
                phase_diff[i].create(size, CV_32FC2); // will have cos and sin in channels
                phase[i].create(size, CV_32FC2); // will have cos and sin in channels
                state[i].create(size, CV_MAKETYPE(CV_32F, N->ival[0] * 2 * 2));
                phase_filt[i].create(size, CV_32FC2);
                phase[i] = 0.0f; // Initialise all elements to zero
                laplacian[i] = new cv::UMat(size, CV_32FC1);
                laplacian_prev[i] = new cv::UMat(size, CV_32FC1);
                *laplacian_prev[i] = 0.0f;
                riesz[i] = new cv::UMat(size, CV_32FC2);
                riesz_prev[i] = new cv::UMat(size, CV_32FC2);
                *riesz_prev[i] = 0.0f;
            }

            if (outfile->count) {
                vidout.open(outfile->filename[0],
                    cv::VideoWriter::fourcc('X', '2', '6', '4'),
                    fs->ival[0],
                    (pyramid[0]).size(),
                    false);

                if (!vidout.isOpened()) {
                    cout << "Failed opening output video" << endl;
                    exit(1);
                }
            }

            first_run = false;
        }
        
        for (int i = pyr_levels-1; i >= 0; i--) {
            // Construct laplacian pyramid
            cv::pyrUp(pyramid[i+1], pyr_ups[i], pyramid[i].size());
            cv::subtract(pyramid[i], pyr_ups[i], *laplacian[i]);

            // TODO: not efficient, can be done in one custom kernel
            cv::filter2D(*laplacian[i], riesz_chans[i][0], -1, riesz_kernel_x);
            cv::filter2D(*laplacian[i], riesz_chans[i][1], -1, riesz_kernel_y);
            cv::merge(riesz_chans[i], *riesz[i]);
        }
       
        for (int i = 0; i < pyr_levels; i++) {
            // Phase difference and amplitude
            pdaa.apply(*laplacian[i], *laplacian_prev[i],
                       *riesz[i], *riesz_prev[i],
                       amp[i], phase_diff[i]);

            // No phase wrapping needed this way
            cv::add(phase[i], phase_diff[i], phase[i]);

            // Temporal filter phase
            butt.apply(phase[i], state[i], phase_filt[i]);

            // Spatial blur the temporally filtered phase
            cv::GaussianBlur(amp[i], denominator[i], cv::Size(0, 0), sigma->ival[0]);
            cv::add(denominator[i], CL_EPS, denominator[i]);
            cv::merge(std::vector<cv::UMat>(2, denominator[i]), denominator2ch[i]);
            
            cv::merge(std::vector<cv::UMat>(2, amp[i]), amp2ch[i]);
            cv::multiply(phase_filt[i], amp2ch[i], numerator[i]); // ?? phase is accumilating?? maybe after filter dc comp removed
            cv::GaussianBlur(numerator[i], numerator[i], cv::Size(0, 0), sigma->ival[0]);
            cv::divide(numerator[i], denominator2ch[i], phase_filt[i]);
            
            // Amplify and phase shift
            aps.apply(*laplacian[i], *riesz[i], phase_filt[i], *laplacian_amp[i]);
        }

        // Collapse magnified laplacian pyramid
        //for (int i = 0; i < pyr_levels; i++) { laplacian_amp[i] = laplacian[i]; }
        cv::add(pyr_ups[pyr_levels - 1], *laplacian_amp[pyr_levels - 1], pyramid[pyr_levels - 1]);

        for (int i = pyr_levels - 2; i >= 0; i--) {
            cv::pyrUp(pyramid[i + 1], pyr_ups[i], pyramid[i].size());
            cv::add(pyr_ups[i], *laplacian_amp[i], pyramid[i]);
        }

        if (!novis->count) {
            cv::imshow("w", pyramid[0]);

            if (cv::waitKey(20) == 27) {
                cout << "key pressed" << endl;
                break;
            }
        }

        if (outfile->count) {
            cv::Mat grey;
            pyramid[0].convertTo(grey, CV_8UC1, 255.0);
            vidout.write(grey);
        }

        // Pointer swappi'n party, yeah !
        cv::UMat* tmp;
        for (int i = 0; i < pyr_levels; i++) {
            tmp = laplacian_prev[i];
            laplacian_prev[i] = laplacian[i];
            laplacian[i] = tmp;

            tmp = riesz_prev[i];
            riesz_prev[i] = riesz[i];
            riesz[i] = tmp;
        }
    }
    
    cap.release();
    if (outfile->count) {
        vidout.release();
    }

    return 0;
}
