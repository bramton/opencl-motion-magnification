#define _USE_MATH_DEFINES
#define M_PI 3.14159265358979323846

#include <cmath>
#include <complex>
#include <vector>

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>

namespace cv
{
class Buttw {
	public:
		Buttw(unsigned int _N, float _fs, float _flo, float _fup) {
			this->N = _N;
			this->fs = _fs;
			this->flo = _flo;
			this->fup = _fup;
			
			filt_weights = Mat::zeros(1, 3 * N, CV_32FC1);
			calcCoeffs();
			initialise();
		}

		bool apply(InputArray _phase, InputArray _state, OutputArray _out) {
			UMat phase = _phase.getUMat();
			UMat state = _state.getUMat();
			UMat out = _out.getUMat();

		    int idxArg = 0;
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnly(phase));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadWriteNoSize(state));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrWriteOnly(out));
			

			size_t globalsize[] = { (size_t)phase.cols, (size_t)phase.rows, 1 };
			return kernel_apply.run(2, globalsize, NULL, true);
		}

		void calcCoeffs() {
			std::vector<std::complex<double>> pa_lp(N); // Poles of prototype LPF (s-domain)
			std::vector<std::complex<double>> pa_bp(N); // Poles of BPF (s-domain)
			std::vector<std::complex<double>> pz_bp(N); // Poles of BPF (z-domain)

			// Poles of the prototype low-pass Butterworth filter (s-domain)
			for (int k = 0; k < N; k++) {
				double theta = ((2.0 * k + 1.0) * M_PI) / (2.0 * N);
				pa_lp[k].real(-sin(theta));
				pa_lp[k].imag(cos(theta));
			}

			// Pre-warp frequencies to compensate for the non-linearity of the bilinear transform
			double Flo = fs / M_PI * tan(M_PI * flo / fs);
			double Fup = fs / M_PI * tan(M_PI * fup / fs);
			double BW = Fup - Flo; // Hz -3 dB bandwidth
			double F0 = sqrt(Flo * Fup); // Hz geometric mean frequency

			// Transform the prototype LPF poles to BPF poles, still in s-domain
			// Conjugates are not computed
			for (int k = 0; k < N; k++) {
				std::complex<double> alpha = (BW / (2.0 * F0)) * pa_lp[k];
				std::complex<double> beta = std::sqrt(std::complex<double>(1.0, 0.0) - std::pow((BW / (2.0 * F0)) * pa_lp[k], 2));
				beta *= std::complex<double>(0.0f, 1.0f);
				pa_bp[k] = (alpha + beta) * 2.0 * M_PI * F0;
			}
			// Bilinear transform poles from s-domain to z-domain
			for (int k = 0; k < N; k++) {
				pz_bp[k] = (pa_bp[k] / (2.0 * fs) + 1.0) / (-pa_bp[k] / (2.0 * fs) + 1.0);
			}
			
			// Denominator coeffs
			double f0 = std::sqrt(flo * fup);
			for (int k = 0; k < N; k++) {
				double a1 = -2.0 * pz_bp[k].real();
				double a2 = std::norm(pz_bp[k]);

				double fn = (2.0 * M_PI * f0) / fs; // Normalised frequency
				std::complex<double> z(cos(fn), sin(fn));
				std::complex<double> H = (std::pow(z, 2) - 1.0) / (std::pow(z, 2) + a1 * z + a2); 
				float gain = 1.0 / abs(H);

				filt_weights.at<float>(0, k * 3 + 0) = gain;// gain;
				filt_weights.at<float>(0, k * 3 + 1) = a1;
				filt_weights.at<float>(0, k * 3 + 2) = a2;
			}

			// Print filter coefficients
			std::cout << "Filter coefficients: " << std::endl;
			std::cout << "b = [1,0,-1] for each cascade" << std::endl;
			for (int k = 0; k < N; k++) {
				std::cout << "cascade " << k << ": a=[1," << filt_weights.at<float>(0, k * 3 + 1) << "," << filt_weights.at<float>(0, k * 3 + 2) << "]";
				std::cout << " gain=" << filt_weights.at<float>(0, k * 3 + 0) << std::endl;
			}

		}

	private:
		cv::ocl::ProgramSource oclsrc;
		ocl::Kernel kernel_apply;
		int N;
		int fs;
		float flo;
		float fup;
		cv::Mat filt_weights;

		void initialise() {
			std::ifstream t("../../../opencl/buttw.cl");
			if (t.fail()) {
				std::cout << "Failed to read buttw.cl" << std::endl;
			}
			std::stringstream buffer;
			buffer << t.rdbuf();
			oclsrc = cv::ocl::ProgramSource(buffer.str());
				
			String opts = cv::format("-D NBIQUADS=%d%s",
				N, ocl::kernelToStr(filt_weights, -1, "FILT_WEIGHTS").c_str());
			kernel_apply.create("buttw_kernel", oclsrc, opts);
		}
	};

}

