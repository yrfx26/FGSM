# from mindspore import load_checkpoint, load_param_into_net
#
# param_dict = load_checkpoint("checkpoint_lenet-5_1875.ckpt")
# load_param_into_net(network, param_dict)
#
# def forward_fn(inputs, targets):
#     out = network(inputs)
#     loss = net_loss(out, targets)
#     return loss
#
#
# from mindspore import grad
#
# grad_fn = grad(forward_fn, 0)
#
# def generate(inputs, labels, eps):
#     # 实现FGSM
#     gradient = grad_fn(inputs, labels)
#     # 产生扰动
#     perturbation = eps * ops.sign(gradient)
#     # 生成受到扰动的图片
#     adv_x = inputs + perturbation
#     return adv_x
#
#
# def batch_generate(inputs, labels, eps, batch_size):
#     # 对数据集进行处理
#     arr_x = inputs
#     arr_y = labels
#     len_x = len(inputs)
#     batches = int(len_x / batch_size)
#     res = []
#     for i in range(batches):
#         x_batch = arr_x[i * batch_size: (i + 1) * batch_size]
#         y_batch = arr_y[i * batch_size: (i + 1) * batch_size]
#         adv_x = generate(x_batch, y_batch, eps=eps)
#         res.append(adv_x)
#     adv_x = ops.concat(res)
#     return adv_x
#
#
# from mindspore import grad
#
# grad_fn = grad(forward_fn, 0)
#
# def generate(inputs, labels, eps):
#     # 实现FGSM
#     gradient = grad_fn(inputs, labels)
#     # 产生扰动
#     perturbation = eps * ops.sign(gradient)
#     # 生成受到扰动的图片
#     adv_x = inputs + perturbation
#     return adv_x
#
#
# def batch_generate(inputs, labels, eps, batch_size):
#     # 对数据集进行处理
#     arr_x = inputs
#     arr_y = labels
#     len_x = len(inputs)
#     batches = int(len_x / batch_size)
#     res = []
#     for i in range(batches):
#         x_batch = arr_x[i * batch_size: (i + 1) * batch_size]
#         y_batch = arr_y[i * batch_size: (i + 1) * batch_size]
#         adv_x = generate(x_batch, y_batch, eps=eps)
#         res.append(adv_x)
#     adv_x = ops.concat(res)
#     return adv_x
#
#
# import mindspore as ms
#
# advs = batch_generate(test_images, true_labels, batch_size=32, eps=0.0)
#
# adv_predicts = model.predict(advs).argmax(1)
# accuracy = ops.equal(adv_predicts, true_labels).astype(ms.float32).mean()
# print(accuracy)
#
#
# advs = batch_generate(test_images, true_labels, batch_size=32, eps=0.5)
#
# adv_predicts = model.predict(advs).argmax(1)
# accuracy = ops.equal(adv_predicts, true_labels).astype(ms.float32).mean()
# print(accuracy)
#
#
#
# import matplotlib.pyplot as plt
# %matplotlib inline
#
# adv_examples = advs[:10].transpose(0, 2, 3, 1)
# ori_examples = test_images[:10].transpose(0, 2, 3, 1)
#
# plt.figure(figsize=(10, 3), dpi=120)
# for i in range(10):
#     plt.subplot(3, 10, i + 1)
#     plt.axis("off")
#     plt.imshow(ori_examples[i].squeeze().asnumpy())
#     plt.subplot(3, 10, i + 11)
#     plt.axis("off")
#     plt.imshow(adv_examples[i].squeeze().asnumpy())
# plt.show()