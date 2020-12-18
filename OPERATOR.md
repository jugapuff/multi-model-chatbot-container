# OPERATOR

## 소개
* 이 설명서는 운영자가 이 프로그램을 운용하기 위한 설명서입니다.

## 사전작업
* Ubuntu 18.04.6 LTS 에서 작동합니다.
* nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 와 호환가능한 GPU와 그래픽 드라이버가 필요합니다

## 설치 및 사용법
* Nvidia Docker가 설치된 환경이 필요합니다
* Dockerfile이 위치한 경로에서 `docker image build . -t <image_name>`
* image build가 완료되면 `docker container run -p 80:80 -d --gpus device=0 --name <contianer_name>  <image_name>` 다음과 같은 명령어로 컨테이너를 run합니다.

## 문제해결
* 문제가 발생시 프로그램을 종료하고 다시 실행합니다.
