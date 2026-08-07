import matplotlib.pyplot as plt



def visualize_data(N,stoi):

    itos = {i:s for s, i in stoi.items()}

    plt.figure(figsize=(16,16))
    plt.imshow(N,cmap='Blues')
    for i in range(27):
        for j in range(27):
            ch_string = itos[i] + itos[j]
            plt.text(j, i , ch_string, ha="center", va="bottom", color="gray")
            plt.text(j, i , N[i,j].item(), ha="center", va="top", color="gray")
    plt.axis('off')

    plt.show()
