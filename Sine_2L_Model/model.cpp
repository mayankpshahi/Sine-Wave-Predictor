#include <math.h>
#include "sine_2L_model_data.h"

float pgm_read_float_near(float *fp){   
  return *fp;
}

inline float W1(int i) {
    return pgm_read_float_near(W1_data + i);
}

inline float b1(int i) {
    return pgm_read_float_near(b1_data + i);
}

inline float W2(int i, int j) {
  return pgm_read_float_near(W2_data + i*DENSE2_SIZE + j);  
}

inline float b2(int i) {
    return pgm_read_float_near(b2_data + i);
}

inline float W3(int i) {
    return pgm_read_float_near(W3_data + i);
}

inline float b3() {
    return pgm_read_float_near(b3_data);
}

inline float relu(float x) {
    return fmaxf(0.0f, x); 
}

float predict(float x) {
    // The activations of the first layer are small enough to store
    // on the stack (16 floats = 64 bytes).
    float h1[DENSE1_SIZE];
    float h2[DENSE2_SIZE];

    // First dense layer. Since there is only one input neuron, we don't need 
    // to perform a full-blown matrix multiply.
    for (int i = 0; i < DENSE1_SIZE; ++i) {
        h1[i] = relu(x * W1(i) + b1(i));
    }

    // Second dense layer.
    for (int i = 0; i < DENSE2_SIZE; ++i) { 
        // Perform a dot product of the incoming activation vector with each 
        // row of the W2 matrix.
        h2[i] = 0.0;
        for (int j = 0; j < DENSE1_SIZE; ++j) {
            h2[i] += h1[j] * W2(j, i);    
        }
        h2[i] = relu(h2[i] + b2(i));
    }

    // Final dense layer.
    float y(0.0f);    
    for (int i = 0; i < DENSE2_SIZE; ++i) { 
        y += h2[i] * W3(i);
    }
    return y + b3();
    
} // end of predict()
