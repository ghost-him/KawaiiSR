# 这个用于查询timm中有哪些模型可以用
import timm
# 查找所有名称中包含 "swin" 的模型
resnet_models = timm.list_models('*swinv2*')
print(f"找到了 {len(resnet_models)} 个 swim 类型的模型。")
print("模型:", resnet_models[:])
