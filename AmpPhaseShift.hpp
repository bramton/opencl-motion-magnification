#pragma once
//#include "precomp.hpp"
//#include <fstream>
//#include <sstream>
//#include <iostream>
#include <opencv2/core/ocl.hpp>

namespace cv
{
class AmpPhaseShift {
	public:
		AmpPhaseShift(unsigned int alpha):alpha(alpha) {
			initialise();
		}

		bool apply(InputArray _lap, InputArray _riesz, InputArray _phase_filt,
			       OutputArray _lap_amp) {

			UMat lap = _lap.getUMat();
			UMat riesz = _riesz.getUMat();
			UMat phase_filt = _phase_filt.getUMat();

			UMat lap_amp = _lap_amp.getUMat();

			int idxArg = 0;
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnly(lap));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnlyNoSize(riesz));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(phase_filt));

			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrWriteOnly(lap_amp));

			size_t globalsize[] = { (size_t)lap.cols, (size_t)lap.rows, 1 };
			return kernel_apply.run(2, globalsize, NULL, true);
		}

	private:
		cv::ocl::ProgramSource oclsrc;
		ocl::Kernel kernel_apply;
		unsigned int alpha; // Amplification factor

		void initialise() {
			std::ifstream t("../../../opencl/amp_phase_shift.cl");
			if (t.fail()) {
				std::cout << "Failed to read amp_phase_shift.cl" << std::endl;
			}
			std::stringstream buffer;
			buffer << t.rdbuf();
			oclsrc = cv::ocl::ProgramSource(buffer.str());

			String opts = cv::format("-D ALPHA=%d", alpha);

			kernel_apply.create("aps_kernel", oclsrc, opts);
		}
	};

}

