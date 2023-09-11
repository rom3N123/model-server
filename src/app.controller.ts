import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { AppService } from './app.service';
import { FormDataRequest, MemoryStoredFile } from 'nestjs-form-data';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { IMAGE_HEIGHT, IMAGE_WIDTH, model } from './model';
import { mkdir, readFile, readdir, writeFile } from 'fs/promises';
import { readFileSync } from 'fs';

export type TrainDto = {
  modelName: string;
  modelDescription: string;
  item1Name: string;
  item2Name: string;
  item1Images: MemoryStoredFile[];
  item2Images: MemoryStoredFile[];
  epochs: string;
};

export type StartDto = {
  images: MemoryStoredFile[];
};

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Post('models/:modelId/start')
  @FormDataRequest()
  async startModel(
    @Param('modelId') modelId: string,
    @Body() { images }: StartDto,
  ) {
    const modelPath = `./models/${modelId}`;
    const model = await tf.loadLayersModel(`file://${modelPath}/model.json`);
    const items = JSON.parse(
      await readFile(`${modelPath}/items.json`, {
        encoding: 'utf-8',
      }),
    );

    const processedImages = images.map((file) => {
      const decodedImage = tf.node.decodeImage(file.buffer);
      const resizedImage = tf.image.resizeBilinear(decodedImage, [
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
      ]);
      return resizedImage.div(tf.scalar(255.0)).expandDims();
    });

    const results = await model.predict(processedImages);

    // @ts-expect-error hello
    const classNameIndex = results.argMax(-1).dataSync()[0];
    const classLabel = items[classNameIndex]?.name || 'Неизвестный класс';

    return classLabel;
  }

  @Get('models')
  async getModels() {
    const ids = await readdir('./models');
    const models: { id: string; name: string; type: 'photo' | 'video' }[] =
      ids.map((id) => {
        const { name } = JSON.parse(
          readFileSync(`./models/${id}/meta.json`, {
            encoding: 'utf-8',
          }),
        );

        return {
          id,
          name,
          type: 'photo',
        };
      });

    return models;
  }

  @Post('model/train')
  @FormDataRequest()
  async train(
    @Body()
    {
      epochs,
      modelName,
      item1Name,
      item2Name,
      modelDescription,
      item1Images,
      item2Images,
    }: TrainDto,
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

    await model.fit(tf.concat(images), labels, {
      batchSize: BATCH_SIZE,
      epochs: Number(epochs),
      shuffle: true,
    });

    const modelId = Date.now().toString();

    const modelFolderPath = `./models/${modelId}`;

    const meta = JSON.stringify({
      name: modelName,
      description: modelDescription,
    });

    const classNames = JSON.stringify([
      {
        name: item1Name,
      },
      { name: item2Name },
    ]);

    await mkdir(modelFolderPath, { recursive: true });
    await writeFile(`${modelFolderPath}/meta.json`, meta);
    await writeFile(`${modelFolderPath}/items.json`, classNames);
    model.save(`file://${modelFolderPath}`);
  }
}
