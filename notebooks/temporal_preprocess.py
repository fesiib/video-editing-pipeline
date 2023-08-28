import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
from chatbot import *
from load_internvideo import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from simplet5 import SimpleT5
from models.grit_model import DenseCaptioning
bot = ConversationBot()
image_size = 384
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])


# define model
model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
model.eval()
model = model.to(device)
print("[INFO] initialize caption model success!")

model_T5 = SimpleT5()
if torch.cuda.is_available():
    model_T5.load_model(
        "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=True)
else:
    model_T5.load_model(
        "t5", "./pretrained_models/flan-t5-large-finetuned-openai-summarize_from_feedback", use_gpu=False)
print("[INFO] initialize summarize model success!")
# action recognition
intern_action = load_intern_action(device)
trans_action = transform_action()
topil =  T.ToPILImage()
print("[INFO] initialize InternVideo model success!")

dense_caption_model = DenseCaptioning(device)
dense_caption_model.initialize_model()
print("[INFO] initialize dense caption model success!")

def inference(video_path, input_tag, progress=gr.Progress()):
    data = loadvideo_decord_origin(video_path)
    progress(0.2, desc="Loading Videos")

    # InternVideo
    action_index = np.linspace(0, len(data)-1, 8).astype(int)
    tmp,tmpa = [],[]
    for i,img in enumerate(data):
        tmp.append(transform(img).to(device).unsqueeze(0))
        if i in action_index:
            tmpa.append(topil(img))
    action_tensor = trans_action(tmpa)
    TC, H, W = action_tensor.shape
    action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        prediction = intern_action(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]

    # dense caption
    dense_caption = []
    dense_index = np.arange(0, len(data)-1, 5)
    original_images = data[dense_index,:,:,::-1]
    with torch.no_grad():
        for original_image in original_images:
            dense_caption.append(dense_caption_model.run_caption_tensor(original_image))
        dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])
    
    # Video Caption
    image = torch.cat(tmp).to(device)   
    
    model.threshold = 0.68
    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        input_tag_list = None
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',',' | '))
    with torch.no_grad():
        caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
        progress(0.6, desc="Watching Videos")
        frame_caption = ' '.join([f"Second {i+1}:{j}.\n" for i,j in enumerate(caption)])
        if input_tag_list == None:
            tag_1 = set(tag_predict)
            tag_2 = ['none']
        else:
            _, tag_1 = model.generate(image,tag_input = None, max_length = 50, return_tag_predict = True)
            tag_2 = set(tag_predict)
        progress(0.8, desc="Understanding Videos")
        synth_caption = model_T5.predict('. '.join(caption))
    print(frame_caption, dense_caption, synth_caption)

    del data, action_tensor, original_image, image,tmp,tmpa
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return ' | '.join(tag_1),' | '.join(tag_2), frame_caption, dense_caption, synth_caption[0], gr.update(interactive = True), prediction

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])


for video_path in os.listdir('clips'):
    model_tag_output, user_tag_output, image_caption_output, dense_caption_output,video_caption_output, chat_video, loadinglabel = inference(video_path,input_tag='none')

    

