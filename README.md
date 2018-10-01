# AttnGAN

Имплементация Pytorch модели AttnGAN для генерации изображений из текстового описания. Основана на оригинальной статье https://arxiv.org/abs/1711.10485 и репозитории авторов https://github.com/taoxugit/AttnGAN

### Используются

* Python 3.6
* Pytorch 0.4.1
* python-dateutil
* easydict
* pandas
* torchfile
* nltk
* scikit-image
        
### Данные

* Метаданные для датасета <a href="https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ">birds</a> - извлечь в папку data/
* Изображения для датасета <a href="http://www.vision.caltech.edu/visipedia/CUB-200-2011.html">birds</a> - извлечь в папку data/birds/

### Обучение

Предобучение DAMSM:
    
    python pretrain_DAMSM.py --cfg cfg/damsm_bird.yml --gpu 0
    
Результаты предобучения записываются в папку output. После завершения - выбранные файлы энкодеров необходимо поместить в папку DAMSMencoders, и прописать к ним путь в конфиге attn_bird.yml 
    
Обучение AttnGAN:
    
    python train.py --cfg cfg/attn_bird.yml --gpu 0

Результаты обучения записываются в папку output. После завершения - выбранные файлы моделей поместить в папку models, и прописать к ним путь в конфиге eval_bird.yml    

    
### Запуск
    
    python eval.py --cfg cfg/eval_bird.yml --gpu 0
    
Сгенерированные изображения помещаются в папку results. По умолчанию, генерация ведется по фразам из data/birds/example_captions.txt  

Также доступен Jupyter Notebook для тестирования - notebooks/eval.ipynb    
  
### Примеры работы

<img src="https://github.com/aleksey-egorov/attngan/blob/master/images/try1.png">    
<img src="https://github.com/aleksey-egorov/attngan/blob/master/images/try2.png">    
    
### Ссылки    

* <a href='https://arxiv.org/abs/1711.10485'>AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks</a>, 2018 - Tao Xu, Pengchuan Zhang, Qiuyuan Huang, Han Zhang, Zhe Gan, Xiaolei Huang, Xiaodong He



