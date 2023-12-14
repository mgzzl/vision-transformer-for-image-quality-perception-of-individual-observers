from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from misc.helpers import prev_img, prev_img_gray, trans_norm2tensor
from model.recorder import Recorder
import seaborn as sns

def normalize_attention_2d(attentions):
    # attentions: numpy array of shape (num_heads, height, width)
    
    # Normalize the attention maps to the range [0, 1]
    attentions_min = np.min(attentions)
    attentions_max = np.max(attentions)

    normalized_attentions = (attentions - attentions_min) / (attentions_max - attentions_min)
    return normalized_attentions

def visualize_attention(model, img, patch_size, device):
    model = Recorder(model).to(device)
    img = img.to(device)  
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size
    
    outputs , attentions = model(img)
    _, preds = torch.max(outputs, dim=1)
    # print(attentions.shape) # torch.Size([1, 6, 16, 257, 257])
    nh = attentions.shape[2]  # number of heads
    nl = attentions.shape[1]  # number of layers
    # print(f"Number of Layers: {nl}")
    # print(f"Number of Heads: {nh}")

    atts = torch.zeros(nl, nh, attentions.shape[-1]-1 , attentions.shape[-1]-1)  # Initialize the result tensor

    for i in range(nl):
        # print(f"Before:{att.shape}")
        att = attentions[0,i,:,0,1:].reshape(nh, -1)
        # print(f"Reshape1:{att.shape}")

        att = att.reshape(nh, w_featmap, h_featmap)
        # print(f"Reshape2:{att.shape}")

        att = nn.functional.interpolate(att.unsqueeze(
        0), scale_factor=patch_size, mode="bilinear")[0]

        atts[i, :, :, :] = att
    atts = atts.cpu().numpy()

    model.clear()
    return preds.cpu().numpy(), atts

def visualize_predict_all_layers_and_heads(model, img, img_size, patch_size, device):
    img_pre = trans_norm2tensor(img, img_size)
    _, attention = visualize_attention(model, img_pre, patch_size, device)
    plot_attention_per_layer_and_heads(img, attention)

def plot_attention_per_layer_and_heads(img, attention):
    n_heads = attention.shape[1]
    n_layers = attention.shape[0]
    image_size = 256

    img_pre = prev_img(img, image_size)
    img_gray = prev_img_gray(img, image_size)

    fig, axes = plt.subplots(n_heads + 1, n_layers + 1, figsize=(20, 50))

    for ax in axes.flat:
        ax.axis('off')

    # Originalbild
    axes[0, 0].imshow(img_pre)
    axes[0, 0].set_title("Original Image")

    for head_idx in range(n_heads+1):
        for layer_idx in range(n_layers):
            ax = axes[head_idx, layer_idx + 1]
            if head_idx == 0:
                layer_mean = np.mean(attention[layer_idx], axis=0)
                layer_mean_norm = normalize_attention_2d(layer_mean)
                ax.imshow(img_gray, cmap='gray')
                sns.heatmap(layer_mean_norm, cmap="inferno", alpha=0.7, ax=ax)
                # ax.imshow(layer_mean_norm, cmap='YlOrRd', alpha=0.9)
                ax.set_title(f"Layer {layer_idx + 1}: Head Mean")
            else:           
                head = attention[layer_idx][head_idx-1]
                head_norm = normalize_attention_2d(head)
                ax.imshow(img_gray, cmap='gray')
                sns.heatmap(head_norm, cmap="inferno", alpha=0.7, ax=ax)
                # ax.imshow(head_norm, cmap='YlOrRd', alpha=0.9)
                ax.set_title(f"Layer {layer_idx + 1}: Head {head_idx}")

    plt.show()