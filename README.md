Real time interactive streaming digital human， realize audio video synchronous dialogue. It can basically achieve commercial effects.  
Интерактивный потоковый цифровой человек в режиме реального времени, реализующий синхронный аудио- и видеодиалог. 
В принципе можно достичь коммерческого эффекта

[ernerf - model](https://www.bilibili.com/video/BV1PM4m1y7Q2/)  [musetalk - model](https://www.bilibili.com/video/BV1gm421N7vQ/)  [wav2lip - model](https://www.bilibili.com/video/BV1Bw4m1e74P/)

## Чтобы избежать путаницы с трехмерными цифровыми людьми, оригинальный проект metahuman-stream Переименован в Livetalking，Исходный адрес ссылки по-прежнему доступен.

## News
- 2024.12.8 Улучшение многопоточности, видеопамять не увеличивается с числом параллельных процессов
- 2024.12.21 wav2lip、добавлено в musetalk Предварительно прогрейте модель, чтобы решить проблему запаздывания во время первого вывода. @heimaojinzhangyz
- 2024.12.28 Добавена цифровая модель человека Ultralight-Digital-Human。 @lijihua2017
- 2025.2.7 fish-speech добавлена в tts
- 2025.2.21 добавлена wav2lip256 - Модель с открытым исходным кодом

## Features
1. Поддержка нескольких цифровых моделей человека: ernerf、musetalk、wav2lip、Ultralight-Digital-Human
2. Поддержка клонирования голоса
3. Поддержка цифровых пользователей, которых прерывают во время разговора
4. Поддержка сшивания видео всего тела
5. Поддержка rtmp и webrtc
6. Поддержка редактирования видео: воспроизведение пользовательских видео, когда вы не говорите
7. Поддержка сразу нескольких цифровых моделей одновременно 

## 1. Installation

Tested on Ubuntu 20.04, Python3.10, Pytorch 1.12 and CUDA 11.3

### 1.1 Install dependency

```bash
conda create -n nerfstream python=3.10
conda activate nerfstream
#Если версия cuda не 11.3 (запустите nvidia-smi, чтобы подтвердить версию), согласно <https://pytorch.org/get-started/previous-versions/>Установите соответствующую версию pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
#Если вам необходимо обучить модель ernerf, установите следующую библиотеку
# pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# pip install tensorflow-gpu==2.8.0
# pip install --upgrade "protobuf<=3.20.1"
``` 
Часто задаваемые вопросы по установке [FAQ](https://livetalking-doc.readthedocs.io/en/latest/faq.html)  
linux cuda: Для создания среды обратитесь к этой статье. https://zhuanlan.zhihu.com/p/674972886


## 2. Quick Start
- Скачать модель
Облако Baidu <https://pan.baidu.com/s/1yOsQ06-RIDTJd3HFCw4wtA> пароль: ltua  
GoogleDriver <https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing>  
Скопируйте wav2lip256.pth в папку models этого проекта и переименуйте его в wav2lip.pth;  
Разархивируйте wav2lip256_avatar1.tar.gz и скопируйте всю папку в этот проект data/avatars
- Запуск  
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1  
Открыть в браузере http://serverip:8010/webrtcapi.html , Первая точка ‘start',Воспроизведите цифровое человеческое видео; затем введите любой текст в текстовое поле и отправьте. Цифровой человек читает этот текст

<font color=red> Серверу необходимо открыть порты tcp:8010; udp:1-65536 </font> 

Если вам нужна коммерческая модель wav2lip высокой четкости, вы можете связаться со мной, чтобы приобрести ее

Если вы не можете получить доступ к huggingface, перед запуском
```
export HF_ENDPOINT=https://hf-mirror.com
``` 


## 3. More Usage
Инструкции: <https://livetalking-doc.readthedocs.io/>
  
## 4. Docker Run  
Никакой предварительной установки не требуется, просто запустите
```
docker run --gpus all -it --network=host --rm registry.cn-beijing.aliyuncs.com/codewithgpu2/lipku-metahuman-stream:vjo1Y6NJ3N
```
Код находится в /root/metahuman-stream. Сначала извлеките последний код с помощью git pull, затем выполните ту же команду, что и в шагах 2 и 3.

Предоставьте следующие изображения
- autodl-Зеркало: <https://www.codewithgpu.com/i/lipku/metahuman-stream/base>   
[autodl-Учебник](https://livetalking-doc.readthedocs.io/en/latest/autodl/README.html)
- ucloud-Зеркало: <https://www.compshare.cn/images-detail?ImageID=compshareImage-18tpjhhxoq3j&referral_code=3XW3852OBmnD089hMMrtuU&ytag=GPU_livetalking1.3>  
Любой порт можно открыть без отдельного развертывания службы srs.
[ucloud-Учебник](https://livetalking-doc.readthedocs.io/en/latest/ucloud/ucloud.html) 


## 5. TODO
- [x] Добавить chatgpt для реализации цифрового человеческого диалога
- [x] Клонирование голоса
- [x] Когда цифровой человек отключен, вместо него будет использоваться видео
- [x] MuseTalk
- [x] Wav2Lip
- [x] Ultralight-Digital-Human

---
Если этот проект был вам полезен, пожалуйста, поставьте ему звездочку. Друзья, которым это интересно, также могут поработать вместе над улучшением проекта.
* Планета Знаний: https://t.zsxq.com/7NMyO Собирайте высококачественные общие проблемы, передовой опыт и решения проблем 
* Публичный аккаунт WeChat: Цифровые технологии человека
![](https://mmbiz.qpic.cn/sz_mmbiz_jpg/l3ZibgueFiaeyfaiaLZGuMGQXnhLWxibpJUS2gfs8Dje6JuMY8zu2tVyU9n8Zx1yaNncvKHBMibX0ocehoITy5qQEZg/640?wxfrom=12&tp=wxpic&usePicPrefetch=1&wx_fmt=jpeg&amp;from=appmsg)  

