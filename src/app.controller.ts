import { Body, Controller, Post } from '@nestjs/common';
import { AppService } from './app.service';
import { FormDataRequest, MemoryStoredFile } from 'nestjs-form-data';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { IMAGE_HEIGHT, IMAGE_WIDTH, model } from './model';

export type TrainDto = {
  modelName: string;
  item1Name: string;
  item1Images: MemoryStoredFile[];
  item2Name: string;
  item2Images: MemoryStoredFile[];
  epochs: number;
};

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Post('model/train')
  @FormDataRequest()
  async train(
    @Body()
    { epochs, item1Images, item1Name, item2Images, item2Name }: TrainDto,
  ) {
    const item1ProcessedImages = item1Images.map((file) => {
      const decodedImage = tf.node.decodeImage(file.buffer);
      const resizedImage = tf.image.resizeBilinear(decodedImage, [
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
      ]);
      return resizedImage.div(tf.scalar(255.0)).expandDims();
    });

    const item2ProcessedImages = item2Images.map((file) => {
      const decodedImage = tf.node.decodeImage(file.buffer);
      const resizedImage = tf.image.resizeBilinear(decodedImage, [
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
      ]);
      return resizedImage.div(tf.scalar(255.0)).expandDims();
    });

    const images = item1ProcessedImages.concat(item2ProcessedImages);
    const labels = tf.tensor2d(
      // @ts-expect-error hello world!
      Array.from({ length: item1ProcessedImages.length })
        .fill([1, 0])
        .concat(
          Array.from({ length: item2ProcessedImages.length }).fill([0, 1]),
        ),
    );

    const BATCH_SIZE = 8;
    const NUM_EPOCHS = 10;

    await model.fit(tf.concat(images), labels, {
      batchSize: BATCH_SIZE,
      epochs: NUM_EPOCHS,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log({ epoch, logs });
        },
      },
    });
  }
}
