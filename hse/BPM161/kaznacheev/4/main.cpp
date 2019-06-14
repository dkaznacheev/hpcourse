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

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        program.build(devices);

        int N;
        fin >> N;

        // create a message to send to kernel
        size_t const block_size = 1024;
        std::vector<double> input(N);
        std::vector<double> output(N, 0);
        for (size_t i = 0; i < N; ++i) {
            fin >> input[i];
        }
        fin.close();

        int total_block_size = N;
        while (total_block_size % block_size) {
            total_block_size++;
        }

        int N_blocks = total_block_size / block_size;

        // allocate device buffer to hold message
        cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * N);
        cl::Buffer dev_block_sum(context, CL_MEM_READ_WRITE, sizeof(double) * N_blocks);
        cl::Buffer dev_new_block_sum(context, CL_MEM_READ_WRITE, sizeof(double) * N_blocks);
        cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(double) * N);
        cl::Buffer dev_small(context, CL_MEM_WRITE_ONLY, sizeof(double));

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * N, &input[0]);

        queue.finish();

        cl::Kernel kernel_b(program, "scan_blelloch");
        cl::KernelFunctor scan_b(kernel_b, queue, cl::NullRange, cl::NDRange(total_block_size), cl::NDRange(block_size));

        cl::Kernel kernel_add(program, "add");
        cl::KernelFunctor add(kernel_add, queue, cl::NullRange, cl::NDRange(total_block_size), cl::NDRange(block_size));

        cl::Event event = scan_b(dev_input, dev_output, dev_block_sum, cl::__local(sizeof(double) * block_size), N);

        event.wait();
        cl_ulong start_time = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

        event = scan_b(dev_block_sum, dev_new_block_sum, dev_small, cl::__local(sizeof(double) * block_size), N_blocks);
        event.wait();

        event = add(dev_output, dev_new_block_sum, N);
        event.wait();

        cl_ulong end_time   = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        cl_ulong elapsed_time = end_time - start_time;

        std::cout << std::setprecision(2) << "Total time: " << elapsed_time / 1000000.0 << " ms" << std::endl;

        queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * N, &output[0]);

        for (int v: output) {
            fout << v << " ";
        }
        fout.close();

    }
    catch (cl::Error &e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}