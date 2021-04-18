#include <math.h>

extern float predict(float x);

#define BUILT_IN_LED 2    // For ESP32

int inference_count = 0;     // Perform Inference using the model with values of x ranging from 0 to 2pi Radians
const int kInferencesPerCycle = 1000;  // Number of samples in one cycle 
const float kXrange = 2*3.14159265359f;   // Full sine wave not 2pi

void setup() {
    Serial.begin(115200); 
    delay(500);
    
    Serial.println("Actual Predicted"); // To print the labels of the values being plotted.
    Serial.print(-1);
    Serial.print("\t");
    Serial.println(+1);
}

void loop() {
    float position = static_cast<float>(inference_count) /
                     static_cast<float>(kInferencesPerCycle);
    float x_val = position * kXrange;

    float y_val = predict(x_val);

    float y_actual = sin(x_val);
    
    Serial.println("Actual Predicted Zero"); // To print the labels of the values being plotted.

    // Enlarge the value so that the plot is enhanced on the Serial Plotter
    y_actual *= 3;
    y_val *= 3;

    Serial.print(y_actual);  // Plot the actual value
    Serial.print("\t");
    Serial.print(y_val);     // Plot the predicted value
    Serial.print("\t");        
    Serial.println(0.0);     // Draw the zero axis

    inference_count += 1;
    
    if (inference_count >= kInferencesPerCycle) 
      inference_count = 0;

    delay(5); // 5 milliseconds of delay between two plots
    
} // end of loop()
