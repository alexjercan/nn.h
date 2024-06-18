#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_WIDTH 128
#define IMAGE_HEIGHT 128
#define IMAGE_CHANNELS 3
#define TRAIN_COUNT 1000

#define CAT_LABEL 1.0
#define DOG_LABEL 0.0
#define EPOCHS 10
#define LEARNING_RATE 0.5

float cat_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};
float dog_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};
float cat_test[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};
float dog_test[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};

float dot_product(const float *xs, const float *ys, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += xs[i] * ys[i];
    }
    return sum;
}

void load_image(const char *filename, float *image) {
    int width, height, channels;

    float *data = stbi_loadf(filename, &width, &height, &channels, IMAGE_CHANNELS);

    stbir_resize_float_linear(data, width, height, 0, image, IMAGE_WIDTH, IMAGE_HEIGHT, 0, STBIR_RGB);

    stbi_image_free(data);
}

void load_dataset(const char *path, const char *label, int offset, int count, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS]) {
    for (int i = offset; i < offset + count; i++) {
        char filename[128] = {0};
        sprintf(filename, "%s/%s.%d.jpg", path, label, i);
        printf("Loading %s\n", filename);
        load_image(filename, train[i]);
    }
}

float rand11() {
    return (rand() / (float)RAND_MAX - 0.5) * 2.0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

typedef struct neural_network {
    float a_0[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    float w_1[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    float b_1;
    float z_1;
    float a_1;
} neural_network;

void nn_init(neural_network *nn) {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; i++) {
        nn->w_1[i] = rand11();
    }
    nn->b_1 = rand11();
}

float nn_loss(float y, float y_hat) {
    return (y - y_hat) * (y - y_hat);
}

float nn_forward(neural_network *nn, float *x) {
    memcpy(nn->a_0, x, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS * sizeof(float));
    nn->z_1  = dot_product(nn->w_1, x, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS) + nn->b_1;
    nn->a_1 = sigmoid(nn->z_1);

    return nn->a_1;
}

void nn_gradient(neural_network *nn, float y_hat, neural_network *grad) {
    // dC/dw_1 = dC/da_1 * da_1/dz_1 * dz_1/dw_1
    float dC_da_1 = 2 * (nn->a_1 - y_hat);
    float da_1_dz_1 = sigmoid(nn->a_1) * (1 - sigmoid(nn->a_1));
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; i++) {
        float dz_1_dw_1 = nn->a_0[i];
        grad->w_1[i] = dC_da_1 * da_1_dz_1 * dz_1_dw_1;
    }
    // dC/db_1 = dC/da_1 * da_1/dz_1 * dz_1/db_1
    float dz_1_db_1 = 1;
    grad->b_1 = dC_da_1 * da_1_dz_1 * dz_1_db_1;
}

void nn_backward(neural_network *nn, neural_network *grad, float learning_rate) {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; i++) {
        nn->w_1[i] -= grad->w_1[i] * learning_rate;
    }
    nn->b_1 -= grad->b_1 * learning_rate;
}

void learn(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat, float learning_rate) {
    for (int i = 0; i < TRAIN_COUNT; i++) {
        neural_network grad_i;
        float y = nn_forward(nn, train[i]);
        nn_gradient(nn, y_hat, &grad_i);
        nn_backward(nn, &grad_i, learning_rate);
    }
}

float compute_loss(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat) {
    float loss = 0;
    for (int i = 0; i < TRAIN_COUNT; i++) {
        float y = nn_forward(nn, train[i]);
        loss += nn_loss(y, y_hat);
    }

    return loss;
}

int compute_true_positive(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat) {
    int count = 0;
    for (int i = 0; i < TRAIN_COUNT; i++) {
        float y = nn_forward(nn, train[i]);
        if (round(y) == y_hat) {
            count++;
        }
    }

    return count;
}

int main() {
    int total = 0;
    int correct = 0;

    srand(time(NULL));
    neural_network nn;

    nn_init(&nn);

    load_dataset("./data/train", "cat", 0, TRAIN_COUNT, cat_train);
    load_dataset("./data/train", "dog", 0, TRAIN_COUNT, dog_train);
    load_dataset("./data/train", "cat", TRAIN_COUNT, TRAIN_COUNT, cat_test);
    load_dataset("./data/train", "dog", 0, TRAIN_COUNT, dog_test);

    total = 2 * TRAIN_COUNT;
    correct = compute_true_positive(&nn, cat_test, CAT_LABEL) + compute_true_positive(&nn, dog_test, DOG_LABEL);
    printf("%f\n", correct / (float)total);

    for (int i = 0; i < EPOCHS; i++) {
        learn(&nn, cat_train, CAT_LABEL, LEARNING_RATE);
        learn(&nn, dog_train, DOG_LABEL, LEARNING_RATE);

        float cat_loss = compute_loss(&nn, cat_train, CAT_LABEL);
        float dog_loss = compute_loss(&nn, dog_train, DOG_LABEL);
        float loss = (cat_loss + dog_loss) / 2.0;
        printf("Loss: %f\n", loss);
    }

    total = 2 * TRAIN_COUNT;
    correct = compute_true_positive(&nn, cat_test, CAT_LABEL) + compute_true_positive(&nn, dog_test, DOG_LABEL);
    printf("%f\n", correct / (float)total);
}
