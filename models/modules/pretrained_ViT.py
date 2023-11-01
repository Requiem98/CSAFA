from libraries import *
import utilities as ut
from timm.models.vision_transformer import _cfg


def vit_base_patch16(pretrained=True, img_size=(224,224), num_classes =1000, **kwargs):
    model = ViT(
        img_size=img_size, patch_size=16, embed_dim=768, num_classes=num_classes, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = vit_base_patch16_224(True).state_dict()

        # resize the positional embedding
        weight = checkpoint['pos_embed']
        ori_size = np.sqrt(weight.shape[1] - 1).astype(int)
        new_size = (img_size[0] // model.patch_embed.patch_size[0], img_size[1] // model.patch_embed.patch_size[1])
        matrix = weight[:, 1:, :].reshape([1, ori_size, ori_size, weight.shape[-1]]).permute((0, 3, 1, 2))
        resize = transforms.Resize(new_size)
        new_matrix = resize(matrix).permute(0, 2, 3, 1).reshape([1, -1, weight.shape[-1]])
        checkpoint['pos_embed'] = torch.cat([weight[:, :1, :], new_matrix], dim=1)
        # change the prediction head if not 1000
        if num_classes != 1000 and num_classes !=0:
            checkpoint['head.weight'] = checkpoint['head.weight'].repeat(5,1)[:num_classes, :]
            checkpoint['head.bias'] = checkpoint['head.bias'].repeat(5)[:num_classes]
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint, strict=False)
    return model 