import streamlit as st
from lib import commons
import torch

def app():
    header=st.container()
    result_all = st.container()
    with header:
        st.subheader("Test whether an area is affected by any natural disaster")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                          "filesize":image_file.size}
            st.write(file_details)

            # To View Uploaded Image
            st.image(commons.load_image(image_file)
                ,width=250
                )
            print("Image file is it showing location?",image_file)
            image_for_model = commons.image_loader(image_file)
            print("Loaded image for model")
        else:
            proxy_img_file="data/joplin-tornado_00000001_post_disaster.png"
            st.image(commons.load_image(proxy_img_file),width=250)
            image_for_model=commons.image_loader(proxy_img_file)
            print("Loaded proxy image for model")

    with result_all:                        
        model_name="squeezenet"
        num_classes = 2        
        feature_extract = False
        # Initialize the model for this run
        model_ft, input_size = commons.initialize_model(model_name, num_classes,
        					 feature_extract, use_pretrained=True)        
        model_state_path="models/squeezenet_10_pre_vs_post_all.pt"
        
        if torch.cuda.is_available():
            model_ft.load_state_dict(torch.load(model_state_path))
        else:
            model_ft.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))
        res=model_ft(image_for_model)
        _, pred = torch.max(res, 1)
        if pred == 0:
            result="No, this area has not been hit by a disaster"
        elif pred == 1:
            result = "Yes, this area has been hit by a disaster"

        st.subheader(result)    