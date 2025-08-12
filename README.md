# DuMo: Dual Encoder Modulation Network for Precise Concept Erasure

🚀 🚀 _AAAI 2025_ | [Arxiv](https://arxiv.org/abs/2501.01125) | 🤗[Models](https://huggingface.co/maplebb/DuMo) 

**Authors**

[Feng Han](https://scholar.google.com.hk/citations?hl=zh-CN&user=oFmRTTkAAAAJ), Kai Chen, Chao Gong, Zhipeng Wei, Jingjing Chen, Yu-Gang Jiang

_Fudan University_

## Run

The Checkpoints of DuMo can be found 🤗[here](https://huggingface.co/maplebb/DuMo). 

* Run `pip install -r requirements.txt` to install the required packages.

Code to train DuMo is coming soon.

Code to generate images of I2P (Nudity):
    
```
python erase/generate_image_nudity.py --model_path /path/to/nudity_checkpoint_dir --save_path images/nudity --sd_path /original_sd1.4
```
Code to generate images of popular artists (Takes "Van Gogh" style erasure as an example):

```
python erase/generate_image_artist_time_new_multi_ft.py --model_path /path/to/van_gogh_checkpoint_dir --save_path images/vangogh --artist_idx 0 --sd_path /original_sd1.4
```
Code to generate images of cartoon characters (Single character erasure):
```
python erase/generate_image_cartoon80_multi_ft.py --model_path /path/to/snoopy_checkpoint_dir --save_path images/snoopy --sd_path /original_sd1.4
```
Code to generate images of cartoon characters (Two characters erasure):
```
python erase/generate_image_cartoon80_multi_ft_sno_mic.py --model_path1 /path/to/snoopy_checkpoint_dir --model_path2 /path/to/mickey_checkpoint_dir --save_path images/snoopy_mickey --sd_path /original_sd1.4
```
Code to generate images of cartoon characters (Three characters erasure):
```
python erase/generate_image_cartoon80_multi_ft_sno_mic_spongebob.py --model_path1 /path/to/snoopy_checkpoint_dir --model_path2 /path/to/mickey_checkpoint_dir --model_path3 /path/to/spongebob_checkpoint_dir --save_path images/snoopy_mickey_spongebob --sd_path /original_sd1.4
```
Finally, evaluate the erasure performance:

LPIPS 

```
python evaluation/lpips_score_artist.py --idx 1 -d0 /path/to/origin_images -d1 /path/to/vangogh_erased_images
(--idx 0 for Picasso, 1 for Van Gogh, 2 for Rembrandt, 3 for Andy Warhol, 4 for Caravaggio)
```
    





## Citation
If you find our work helpful, please leave us a star and cite our paper.
  
  ```
@article{han2025dumo,
  title={DuMo: Dual Encoder Modulation Network for Precise Concept Erasure},
  author={Han, Feng and Chen, Kai and Gong, Chao and Wei, Zhipeng and Chen, Jingjing and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2501.01125},
  year={2025}
}
  ```

## Acknowledgement
Some code is borrowed from [Controlnet](https://github.com/lllyasviel/ControlNet).