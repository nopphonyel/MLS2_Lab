import torch
from torchvision import transforms, models
from PIL import Image

MDL_PATH = 'RESNET101_ADAM_Cervix.pth'
OUTPUT = {
    0: "RISK",
    1: "NORMAL"
}

preprocess = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if __name__ == '__main__':
    print("Loading data...")
    mdl = models.resnet101()
    mdl.fc = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
    mdl.load_state_dict(torch.load(MDL_PATH))
    mdl.eval()
    print("Loaded successfully\n\t> type \'quit\' to exit program")
    while (True):
        in_img_path = input("Specify image path: ")
        if in_img_path.lower() == "quit":
            break

        try:
            in_img = Image.open(in_img_path)
            in_tensor = preprocess(in_img)
            in_batch = in_tensor.unsqueeze(0)
            res = mdl(in_batch)

            softmax_scores = torch.nn.functional.softmax(res[0], dim=0)

            maxval, maxindex = res.max(1)
            # print('RES_value' , res)
            # print('Maximum value', maxval, 'at index', maxindex)
            final_res = maxindex.item()
            print(OUTPUT[final_res])
        except FileNotFoundError:
            print("File not found")
