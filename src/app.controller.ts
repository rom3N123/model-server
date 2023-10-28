import { Body, Controller, Delete, Get, Param, Post } from '@nestjs/common';
import { AppService } from './app.service';
import { FormDataRequest, MemoryStoredFile } from 'nestjs-form-data';
import * as tf from '@tensorflow/tfjs-node';
import { IMAGE_HEIGHT, IMAGE_WIDTH, createModel } from './model';
import { mkdir, readFile, readdir, writeFile, rm } from 'fs/promises';
import { readFileSync } from 'fs';

export type TrainDto = any;

export type StartDto = {
  images: MemoryStoredFile[];
};

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Delete('models/:modelId')
  async deleteModel(@Param('modelId') modelId: string) {
    const modelPath = `./models/${modelId}`;

    await rm(modelPath, { recursive: true });
  }

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

    const results = await Promise.all(
      processedImages.map((image) => model.predict(image)),
    );

    const classNameIndexes = results.map((result) =>
      // @ts-expect-error aaa
      result.argMax(-1).dataSync(),
    );

    const classNames = classNameIndexes.map(
      ([index]) => items[index]?.name || 'Неизвестный класс',
    );

    return classNames;
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
    { classNames, modelName, epochs, ...data }: TrainDto,
  ) {
    const numberOfClassess = classNames.length;

    const processedImages = classNames.map((_, index) => {
      const images = data[`class_${index}_images`];

      return images.map((file) => {
        const decodedImage = tf.node.decodeImage(file.buffer);
        const resizedImage = tf.image.resizeBilinear(decodedImage, [
          IMAGE_WIDTH,
          IMAGE_HEIGHT,
        ]);
        return resizedImage.div(tf.scalar(255.0)).expandDims();
      });
    });

    const labels = processedImages.map((images, index) => {
      const length: number = images.length;
      const labels = Array.from({ length }).fill(
        Array.from({ length: numberOfClassess }, (_, idx) =>
          idx === index ? 1 : 0,
        ),
      ) as number[][];

      return labels;
    });

    const BATCH_SIZE = 8;

    const model = createModel({
      numberOfClassess,
    });

    await model.fit(
      tf.concat(processedImages.flat()),
      tf.tensor2d(labels.flat()),
      {
        batchSize: BATCH_SIZE,
        epochs: Number(epochs),
        shuffle: true,
      },
    );

    const modelId = Date.now().toString();

    const modelFolderPath = `./models/${modelId}`;

    const meta = JSON.stringify({
      name: modelName,
    });

    const classess = JSON.stringify(
      classNames.map((className) => ({ name: className })),
    );

    await mkdir(modelFolderPath, { recursive: true });
    await writeFile(`${modelFolderPath}/meta.json`, meta);
    await writeFile(`${modelFolderPath}/items.json`, classess);
    model.save(`file://${modelFolderPath}`);
  }
}
