import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { NestjsFormDataModule } from 'nestjs-form-data';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [
    ConfigModule.forRoot(),
    NestjsFormDataModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
