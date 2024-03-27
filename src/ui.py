from customtkinter import *
from PIL import Image
from predict import *

app = CTk()
app.geometry("900x700")

def selectfile():
    filename = filedialog.askopenfilename()
    print(filename)
    global image_file
    image_file = filename
    img = Image.open(filename)
    image = CTkImage(light_image=img, dark_image=img, size=(400, 400))
    imLabel = CTkLabel(app, text="", image=image)
    imLabel.place(relx=0.5, rely=0.5, anchor="center")

def classify():
    model = CarBikeClassifier(num_classes=2)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    print(image_file)
    prediction_text = predict_image(model, image_file, device='cpu')

    frame = CTkFrame(master=app, fg_color="white", bg_color="gray", border_color="black", border_width=2)
    frame.place(relx=0.5, rely=0.1, anchor="center")
    output_text = "This is a " + prediction_text
    txt = CTkLabel(master=frame, text=output_text, font=("Arial", 20), fg_color="palegreen3", pady=5, padx=5)
    txt.pack(anchor="s", expand=True, pady=3, padx=3)

button_to_select = CTkButton(master=app, text="Select Image", fg_color="orange", bg_color="lightblue", command=selectfile)
button_to_select.pack(padx=5, pady=5)
button_to_select.place(relx=0.4, rely=0.9, anchor="center")

classify_button = CTkButton(master=app, text="Classify", fg_color="seagreen3", bg_color="palevioletred3", command=classify)
classify_button.place(relx=0.6, rely=0.9, anchor="center")

app.mainloop()
