# Yolo_v3_study

## 정의(Yolo란??)

 YOLO(You Only Look Once)란 deep CNN(Convolutional Neural Network)기반의 모델로, object를 detection하는데 사용된다. 

## YOLO 모델의 특징

- YOLO는 Convolutional Layers(CNN Layer)로 구성된 모델이다.  
- Skip Connection, upsampling layer를 포함해 총 75개의 CNN Layer로 구성되어 있다.
- YOLO 모델에서는 low-feature의 소실을 막기위해 Pooling층을 사용하지 않는다.
- 입력이 항상 일정해야한다. (전처리를 통해 Resize를 하는것이 필요함)
- YOLO 에서는 stride를 통해 downsampling을 한다.

## YOLO 모델의 출력

- YOLO의 prediction은 1x1 CNN layer를 사용해 이뤄진다. (따라서 Feature map이 곧 출력 값)

- 각 Cell은 고정된 숫자의 bounding box를 예측하게 된다.

  ```
  Cell 이란, NxN Feature map에서 한개의 1x1 사이즈의 한 Feature map을 의미한다. 
  ```

- 이때 각 셀의 깊이는 (B x (5 + C)) 의 깊이를 가지고 있다. 

  B : bounding box 수

  5 + C : (bounding box의 x좌표, bounding box의 y좌표, bounding box의 너비, bounding box의 높이 , 신뢰도, 클래스 수 (C))

- Object의 중심이 해당 cell에 있는 경우, feature map의 각 cell이 bounding box를 사용해 object를 예측함
- 이때 중심이 되는 해당 cell은 해당 Object를 책임지고 predict한다.

### Anchor boxes

 anchor boxes는 YOLO 모델에서 예측하는 bounding box와 동의어이다. 

YOLO v3는 3개의 bounding box를 예측한다. 이때 ground truth box(label)와 [IoU](https://ballentain.tistory.com/12) 비교를 통해 값을 결정하게 된다. 

## Prediction

### 중심 좌표

- YOLO 모델은 절대 위치를 예측하지 않음 Cell단위에서 상대적 위치를 예측함.

  즉, 0~1로 정규화된 값(sigmoid function을 사용)으로 중심좌표를 예측함. 

  ```
  ** 0~1의 정규화된 값으로 중심좌표를 예측하는 이유
  -> 해당 cell에서 책임지고 anchors를 예측해야 하는데, 0 미만이거나 1 이상의 값이 되면, 중심의 위치가 다른 cell로 옮겨지게됨
  ```

![img](https://dojinkimm.github.io/assets/imgs/yolo/yolo_part1_2.png)

### 학습 데이터

- [http://detrac-db.rit.albany.edu/Detection](http://detrac-db.rit.albany.edu/Detection)