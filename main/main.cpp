/* Edge Impulse Espressif ESP32 Standalone Inference ESP IDF Example
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Include ----------------------------------------------------------------- */
#include <stdio.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

#include "driver/gpio.h"
#include "sdkconfig.h"

#define LED_PIN GPIO_NUM_3

static const float features[] = {
    0.2274, -6.9073, -9.8593, 0.2274, -6.9073, -9.8593, 0.9146, 3.9959,
    -10.8218, 0.9146, 3.9959, -10.8218, 0.9146, 3.9959, -10.8218, 0.9146,
    3.9959, -10.8218, 0.9146, 3.9959, -10.8218, 0.9146, 3.9959, -10.8218,
    0.9146, 3.9959, -10.8218, 0.9146, 3.9959, -10.8218, 0.9146, 3.9959,
    -10.8218, 0.9146, 3.9959, -10.8218, 0.9146, 3.9959, -10.8218, -0.8308,
    -0.1269, -11.4323, -0.8308, -0.1269, -11.4323, -0.8308, -0.1269, -11.4323,
    -0.8308, -0.1269, -11.4323, -0.8308, -0.1269, -11.4323, -0.8308, -0.1269,
    -11.4323, -0.8308, -0.1269, -11.4323, -0.8308, -0.1269, -11.4323, -0.8308,
    -0.1269, -11.4323, -0.8308, -0.1269, -11.4323, -0.8308, -0.1269, -11.4323,
    -0.1077, 1.4557, -10.9032, -0.1077, 1.4557, -10.9032, -0.1077, 1.4557,
    -10.9032, -0.1077, 1.4557, -10.9032, -0.1077, 1.4557, -10.9032, -0.1077,
    1.4557, -10.9032, -0.1077, 1.4557, -10.9032, -0.1077, 1.4557, -10.9032,
    -0.1077, 1.4557, -10.9032, -0.1077, 1.4557, -10.9032, -0.1077, 1.4557,
    -10.9032, -0.0311, 2.4373, -11.2743, -0.0311, 2.4373, -11.2743, -0.0311,
    2.4373, -11.2743, -0.0311, 2.4373, -11.2743, -0.0311, 2.4373, -11.2743,
    -0.0311, 2.4373, -11.2743, -0.0311, 2.4373, -11.2743, -0.0311, 2.4373,
    -11.2743, -0.0311, 2.4373, -11.2743, -0.0311, 2.4373, -11.2743, -0.0311,
    2.4373, -11.2743, -0.0311, 2.4373, -11.2743, -0.3974, 3.5913, -10.5967,
    -0.3974, 3.5913, -10.5967, -0.3974, 3.5913, -10.5967, -0.3974, 3.5913,
    -10.5967, -0.3974, 3.5913, -10.5967, -0.3974, 3.5913, -10.5967, -0.3974,
    3.5913, -10.5967, -0.3974, 3.5913, -10.5967, -0.3974, 3.5913, -10.5967,
    -0.3974, 3.5913, -10.5967, -0.3974, 3.5913, -10.5967, 0.3567, 3.8738,
    -11.4323, 0.3567, 3.8738, -11.4323, 0.3567, 3.8738, -11.4323, 0.3567,
    3.8738, -11.4323, 0.3567, 3.8738, -11.4323, 0.3567, 3.8738, -11.4323,
    0.3567, 3.8738, -11.4323, 0.3567, 3.8738, -11.4323, 0.3567, 3.8738,
    -11.4323, 0.3567, 3.8738, -11.4323, 0.3567, 3.8738, -11.4323, 0.4501,
    3.6751, -11.5784, 0.4501, 3.6751, -11.5784, 0.4501, 3.6751, -11.5784,
    0.4501, 3.6751, -11.5784, 0.4501, 3.6751, -11.5784, 0.4501, 3.6751,
    -11.5784, 0.4501, 3.6751, -11.5784, 0.4501, 3.6751, -11.5784, 0.4501,
    3.6751, -11.5784, 0.4501, 3.6751, -11.5784, 0.4501, 3.6751, -11.5784,
    -0.1293, 2.1237, -11.1809, -0.1293, 2.1237, -11.1809, -0.1293, 2.1237,
    -11.1809, -0.1293, 2.1237, -11.1809, -0.1293, 2.1237, -11.1809, -0.1293,
    2.1237, -11.1809, -0.1293, 2.1237, -11.1809, -0.1293, 2.1237, -11.1809,
    -0.1293, 2.1237, -11.1809, -0.1293, 2.1237, -11.1809, -0.1293, 2.1237,
    -11.1809, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420,
    -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420,
    -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420,
    -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420, -10.6518, 0.0072, 1.1420,
    -10.6518, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258,
    -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258,
    -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258,
    -11.3174, -0.1006, 1.2258, -11.3174, -0.1006, 1.2258, -11.3174, 0.4741, -0.8835,
    -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835,
    -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835,
    -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835, -11.5951, 0.4741, -0.8835,
    -11.5951, 0.4741, -0.8835, -11.5951
};

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr)
{
  memcpy(out_ptr, features + offset, length * sizeof(float));
  return 0;
}

void print_inference_result(ei_impulse_result_t result) {

    // Print how long it took to perform inference
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

}

extern "C" int app_main()
{
    // Gree LED ESP32-S3-Eye
    gpio_pad_select_gpio(LED_PIN);
    gpio_reset_pin(LED_PIN); 
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);

    ei_sleep(100);

    ei_impulse_result_t result = { nullptr };

    ei_printf("Edge Impulse standalone inferencing (Espressif ESP32)\n");

    if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
        ei_printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
        return 1;
    }

    while (true)
    {
        // blink LED
        gpio_set_level(LED_PIN, 1);

        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;

        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, false /* debug */);
        if (res != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            return res;
        }

        print_inference_result(result);

        gpio_set_level(LED_PIN, 0);
        ei_sleep(1000);
    }
}

