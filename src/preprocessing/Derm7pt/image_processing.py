# def _get_transform_pipeline(use_training_transforms, resol):
#     # resized_resol - resizes to slightly larger to ensure image is large enough before cropping to resol
#     resized_resol = int(resol * 256/224) # 299 * 256/224 = 341.7
#     if use_training_transforms:
#         # print("Using TRAINING transformations:")
#         return transforms.Compose([
#             transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
#             transforms.RandomResizedCrop(resol),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
#             ])
#     # Use LANCZOS resampling for better quality
#     # print("Using VALIDATION/TEST transformations:")
#     return transforms.Compose([
#         transforms.Resize(resized_resol, interpolation=transforms.InterpolationMode.LANCZOS),
#         transforms.CenterCrop(resol),
#         transforms.ToTensor(), # divide by 255, convert from [0,255] -> [0.0,1.0]
#         transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2]) # shifts from [0.0, 1.0] -> [-0.25, 0.25]
#         ])
