import * as tf from '@tensorflow/tfjs-node';

export const IMAGE_WIDTH = 224;
export const IMAGE_HEIGHT = 224;

export type CreateModelConfig = {
  numberOfClassess: number;
};

export const createModel = ({ numberOfClassess }: CreateModelConfig) => {
  const model = tf.sequential();

  model.add(
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, 3],
      filters: 16,
      kernelSize: 3,
      activation: 'relu',
    }),
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  model.add(
    tf.layers.conv2d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
    }),
  );

  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    }),
  );

  model.add(tf.layers.flatten());

  model.add(
    tf.layers.dense({
      units: 64,
      activation: 'relu',
    }),
  );

  model.add(
    tf.layers.dense({
      units: numberOfClassess,
      activation: 'softmax',
    }),
  );

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};
