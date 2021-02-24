#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <stdlib.h>
#include "genann.h"


struct state {
    double pos;
    double slots;
};


void print_arr(const double* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        printf("[%d] %3.2f ", i, arr[i]);
    printf("\n");
}


void evaluate_test(const genann* ann, int step = 5) 
{
    double st[2] = {};
    for (int i = 0; i < 360; i += step) {
        st[0] = i / 360.0;
        st[1] = 1; //(rand() % 3 + 4) / 6.0;
        const double* outputs = genann_run(ann, st);
        printf("%3d ", i);
        print_arr(outputs, 6);
    }
}


int main(int argc, char const *argv[])
{
    // gennan test that learns how to get the player slot based on the player
    // angle in Super Hexagon. 

    // test.txt data has a format of: <angle> <slots> <result slot>

    // all input vectors must be normalized! (lesson learned).

    std::ifstream ifs("test.txt");
    std::vector<state> states;
    std::vector<std::array<double, 6> > results;

    double pos, slots, result;
    while (ifs >> pos >> slots >> result) {
        states.push_back({pos / 360.0, slots / 6.0});
        std::array<double, 6> arr;
        for (int i = 0; i < 6; ++i)
            arr[i] = i == result ? 1 : 0;
        results.push_back(arr);
    }

    ifs.close();

    genann *ann = genann_init(2, 1, 6, 6);

    for (int test = 0; test < 50; ++test) {
        for (int i = 0; i < states.size(); ++i) {
            genann_train(ann, (double*) &states[i], results[i].data(), 0.2);
            // evaluate_test(ann, 30);
        }
    }

    evaluate_test(ann);

    FILE* f = fopen("weights.ann", "w");
    genann_write(ann, f);
    fclose(f);

    genann_free(ann);

    return 0;
}
