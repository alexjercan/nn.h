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
#define TRAIN_COUNT 100
#define HIDDEN_LAYER_SIZE 16

float cat_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};
float dog_train[TRAIN_COUNT][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS] = {0};

float dot_product(const float *xs, const float *ys, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += xs[i] * ys[i];
    }
    return sum;
}

void matrix_vector_product(const float *matrix, const float *vector, float *result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = dot_product(matrix + i * cols, vector, cols);
    }
}

void load_image(const char *filename, float *image) {
    int width, height, channels;

    float *data = stbi_loadf(filename, &width, &height, &channels, IMAGE_CHANNELS);

    stbir_resize_float_linear(data, width, height, 0, image, IMAGE_WIDTH, IMAGE_HEIGHT, 0, STBIR_RGB);

    stbi_image_free(data);
}

void load_dataset(const char *path, const char *label, int count, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS]) {
    for (int i = 0; i < count; i++) {
        char filename[128] = {0};
        sprintf(filename, "%s/%s.%d.jpg", path, label, i);
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
    float weights_1[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS][HIDDEN_LAYER_SIZE];
    float bias_1[HIDDEN_LAYER_SIZE];
    float z_1[HIDDEN_LAYER_SIZE];
    float a_1[HIDDEN_LAYER_SIZE];
    float weights_2[HIDDEN_LAYER_SIZE];
    float bias_2;
    float z_2;
    float a_2;
} neural_network;

void nn_init(neural_network *nn) {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            nn->weights_1[i][j] = rand11();
        }
    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->bias_1[i] = rand11();
    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->weights_2[i] = rand11();
    }
    nn->bias_2 = rand11();
}

float nn_forward(neural_network *nn, float *x) {
    memcpy(nn->a_0, x, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
    matrix_vector_product((float *)nn->weights_1, x, nn->z_1, HIDDEN_LAYER_SIZE, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->a_1[i] = sigmoid(nn->z_1[i] + nn->bias_1[i]);
    }
    nn->z_2 = dot_product(nn->a_1, nn->weights_2, HIDDEN_LAYER_SIZE);
    nn->a_2 = sigmoid(nn->z_2 + nn->bias_2);
    return nn->a_2;
}

float nn_loss(float y_hat, float y) {
    return (y_hat - y) * (y_hat - y);
}

void nn_gradient(const neural_network *nn, float y_hat, neural_network *grad) {
    float dC_da_2 = 2 * (nn->a_2 - y_hat);
    float da_dz_2 = nn->a_2 * (1 - nn->a_2);
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        float dz_dw_2 = nn->a_1[i];
        grad->weights_2[i] = dC_da_2 * da_dz_2 * dz_dw_2;

        float dz_db = 1;
        grad->bias_2 = dC_da_2 * da_dz_2 * dz_db;
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        float dC_da_1 = nn->weights_2[i] * dC_da_2 * da_dz_2;
        float da_dz_1 = nn->a_1[i] * (1 - nn->a_1[i]);
        for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; j++) {
            float dz_dw_1 = nn->a_0[j];
            grad->weights_1[j][i] = dC_da_1 * da_dz_1 * dz_dw_1;
        }

        float dz_db = 1;
        grad->bias_1[i] = dC_da_1 * da_dz_1 * dz_db;
    }
}

void nn_backward(neural_network *nn, const neural_network *grad, float learning_rate) {
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; i++) {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            nn->weights_1[i][j] -= learning_rate * grad->weights_1[i][j];
        }
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->bias_1[i] -= learning_rate * grad->bias_1[i];
    }

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        nn->weights_2[i] -= learning_rate * grad->weights_2[i];
    }

    nn->bias_2 -= learning_rate * grad->bias_2;
}

void run_train(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat, float learning_rate) {
    neural_network grad;
    for (int i = 0; i < TRAIN_COUNT; i++) {
        neural_network grad_i;
        nn_forward(nn, train[i]);
        nn_gradient(nn, y_hat, &grad_i);
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            grad.weights_2[i] += grad_i.weights_2[i];
        }
        grad.bias_2 += grad_i.bias_2;
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; j++) {
                grad.weights_1[j][i] += grad_i.weights_1[j][i];
            }
        }
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            grad.bias_1[i] += grad_i.bias_1[i];
        }

    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        grad.weights_2[i] /= TRAIN_COUNT;
    }
    grad.bias_2 /= TRAIN_COUNT;
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        for (int j = 0; j < IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS; j++) {
            grad.weights_1[j][i] /= TRAIN_COUNT;
        }
    }
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
        grad.bias_1[i] /= TRAIN_COUNT;
    }
    nn_backward(nn, &grad, learning_rate);
}

float compute_loss(neural_network *nn, float train[][IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS], float y_hat) {
    float loss = 0;
    for (int i = 0; i < TRAIN_COUNT; i++) {
        float y = nn_forward(nn, train[i]);
        loss += nn_loss(y, y_hat);
    }

    return loss / TRAIN_COUNT;
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

#define EPOCHS 100
#define LEARNING_RATE 1.0

int main() {
    int total = 0;
    int correct = 0;

    srand(time(NULL));
    neural_network nn;

    nn_init(&nn);

    load_dataset("./data/train", "cat", TRAIN_COUNT, cat_train);
    load_dataset("./data/train", "dog", TRAIN_COUNT, dog_train);

    total = 2 * TRAIN_COUNT;
    correct = compute_true_positive(&nn, cat_train, 1.0) + compute_true_positive(&nn, dog_train, 0.0);
    printf("%f\n", correct / (float)total);

    for (int i = 0; i < EPOCHS; i++) {
        run_train(&nn, cat_train, 1.0, LEARNING_RATE);
        run_train(&nn, dog_train, 0.0, LEARNING_RATE);

        float cat_loss = compute_loss(&nn, cat_train, 1.0);
        float dog_loss = compute_loss(&nn, dog_train, 0.0);

        float loss = (cat_loss + dog_loss) / 2;
        printf("Epoch %d: Loss %f\n", i, loss);
    }

    total = 2 * TRAIN_COUNT;
    correct = compute_true_positive(&nn, cat_train, 1.0) + compute_true_positive(&nn, dog_train, 0.0);
    printf("%f\n", correct / (float)total);
}
