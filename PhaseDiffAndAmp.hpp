#pragma once
//#include "precomp.hpp"
//#include <fstream>
//#include <sstream>
//#include <iostream>
#include <opencv2/core/ocl.hpp>

namespace cv
{
class PhaseDiffAndAmp {
	public:
		PhaseDiffAndAmp() {
			initialise();
		}

		bool apply(InputArray _laplacian, InputArray _laplacian_prev,
			       InputArray _riesz, InputArray _riesz_prev,
			       OutputArray _amp, OutputArray _phase_diff) {
			UMat laplacian = _laplacian.getUMat();
			UMat laplacian_prev = _laplacian_prev.getUMat();

			UMat riesz = _riesz.getUMat();
			UMat riesz_prev = _riesz_prev.getUMat();

			UMat amp = _amp.getUMat();
			UMat phase_diff = _phase_diff.getUMat();

			int idxArg = 0;
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnly(laplacian));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(laplacian_prev));

			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::ReadOnlyNoSize(riesz));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrReadOnly(riesz_prev));

			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrWriteOnly(amp));
			idxArg = kernel_apply.set(idxArg, ocl::KernelArg::PtrWriteOnly(phase_diff));

			size_t globalsize[] = { (size_t)laplacian.cols, (size_t)laplacian.rows, 1 };
			return kernel_apply.run(2, globalsize, NULL, true);
		}

	private:
		cv::ocl::ProgramSource oclsrc;
		ocl::Kernel kernel_apply;

		void initialise() {
			std::ifstream t("../../../opencl/phase_diff_and_amp.cl");
			if (t.fail()) {
				std::cout << "Failed to read phase_diff_and_amp.cl" << std::endl;
			}
			std::stringstream buffer;
			buffer << t.rdbuf();
			oclsrc = cv::ocl::ProgramSource(buffer.str());

			kernel_apply.create("pdaa_kernel", oclsrc, "");
		}
	};

}

