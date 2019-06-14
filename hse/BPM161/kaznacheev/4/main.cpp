#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    std::ifstream fin("input.txt");
    std::ofstream fout("output.txt");

    try {
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
        cl::Context context(devices);

        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        cl::Program program(context, source);
        program.build(devices);

        int N;
        fin >> N;

        size_t const block_size = 1024;
        std::vector<double> input(N);
        std::vector<double> output(N, 0);
        for (size_t i = 0; i < N; ++i) {
            fin >> input[i];
        }
        fin.close();

        int total_blocks = N;
        while (total_blocks % block_size) {
            total_blocks++;
        }

        int block_num = total_blocks / block_size;

        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * N);
        cl::Buffer dev_block_sum(context, CL_MEM_READ_WRITE, sizeof(double) * block_num);
        cl::Buffer dev_new_block_sum(context, CL_MEM_READ_WRITE, sizeof(double) * block_num);
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(double) * N);
        cl::Buffer dev_small(context, CL_MEM_WRITE_ONLY, sizeof(double));
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * N, &input[0]);

        queue.finish();

        cl::Kernel kernel_b(program, "scan");
        cl::KernelFunctor scan_b(kernel_b, queue, cl::NullRange, cl::NDRange(total_blocks), cl::NDRange(block_size));

        cl::Kernel kernel_add(program, "add");
        cl::KernelFunctor add(kernel_add, queue, cl::NullRange, cl::NDRange(total_blocks), cl::NDRange(block_size));

        cl::Event event = scan_b(dev_input, dev_output, dev_block_sum, cl::__local(sizeof(double) * block_size), N);

        event.wait();
        event = scan_b(dev_block_sum, dev_new_block_sum, dev_small, cl::__local(sizeof(double) * block_size), block_num);
        event.wait();

        event = add(dev_output, dev_new_block_sum, N);
        event.wait();

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * N, &output[0]);

        for (int v: output) {
            fout << v << " ";
        }
        fout.close();

    }
    catch (cl::Error &e) {
        std::cerr << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}