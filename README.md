# MotionDetection

## Описание задачи
Motion detection система выделяет, где на видеоряде было движение и выделяет его через bounding box.

## Описание модели
Модель, которую я планирую использовать является представителем Encoder-Decoder моделей. Тем не менее, разительное отличие, 
в перспективе позволяющее получить впечатляющие результаты - context vector'ы, полученные на выходе encoder, передаются на вход
LSTM и лишь затем в Decoder. Это позволяет получить контекст сразу для нескольких кадров, и точно определить движение.
А нейронная сеть, задействованая в процессе, позволит определять, является ли отловленное движение важным (то есть, НЕ шумом),
и лишь затем выделять.

## OpenCV demo
OpenCV demo - это программа, написанная без использования нейронных сетей для демонстрации работы модели. 

## ССЫЛКИ
[ТЗ](https://docs.google.com/document/d/1jicsDc5AsXjXKgJPSu_Y5KjNy98t7uhytXMtv7glpjc/edit?usp=sharing)
