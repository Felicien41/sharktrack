# SharkTrack
This is a package to run a shark detector and tracker model and compute MaxN annotations from it.


[![watrch video](static/video_screenshot.png)](https://drive.google.com/file/d/1b_74wdPXyJPe2P-m1c45jjsV2C5Itr-R/view?usp=sharing)
Click on the image above to watch a demo

## Processing SharkTrack Annotations


📺 **Tutorials**:
You can also follow this documentation using the following video tutorials:

- [Uploading and cleaning detections in VIAME](https://drive.google.com/file/d/16Zw69ELvA1_pBhfcbQsjo1nc_7EBYZl2/view?usp=sharing)
- [Computing MaxN after downloading VIAME-cleaned detections](https://drive.google.com/file/d/1DCT3vCAbAH4T8wTiMjgWUc7-lZEpgz9U/view?usp=drive_link)

This document outlines a simple process to clean the predictions generated by the SharkTrack model and generate a MaxN file.

1. **Familiarise with the Output:** locate the `./output` file, containing the model output. This file contains the following subitems:
    1. `./output.csv` a csv file listing every detection in every frame of every video
    2. `./viame.csv` a csv file listing one detection per tracked shark, for the frame in which it was detected with highest confidence
    3. `./detections` for each tracked shark, this folder shows the frame in which it achieved maximum confidence. As you can see from the image below, the image also shows the other detections, although we are interested only in the highlighted one.
        
        ![Screenshot 2024-03-15 at 16.37.45.png](static/Screenshot_2024-03-15_at_16.37.45.png)
        
2. **Setup Annotations Platform**
    1. Open [VIAME](https://viame.kitware.com/)
    2. Create an account
    3. Click “Upload“ > Add Image Sequence
        
        ![Screenshot 2024-03-15 at 16.45.25.png](static/Screenshot_2024-03-15_at_16.45.25.png)
        
        ![Screenshot 2024-03-15 at 16.45.51.png](static/Screenshot_2024-03-15_at_16.45.51.png)
        
    4. Upload all the images in `./detections`
    5. Click on “annotation file” and upload `viame.csv`
        
        ![Screenshot 2024-03-15 at 16.47.14.png](static/Screenshot_2024-03-15_at_16.47.14.png)
        
    6. Pick a name for the BRUVS analysis

        ![analysis_name.png](static/analysis_name.png)
    7. Confirm upload
3. **Clean Annotations**
    1. Click Launch Annotator
    2. For each frame
        
        ![Screenshot 2024-03-15 at 16.49.38.png](static/Screenshot_2024-03-15_at_16.49.38.png)
        
        1. Identify the track by clicking on the highlighted bounding box
        2. If the detection is valid, insert the shark species
            
            ![Screenshot 2024-03-15 at 16.52.17.png](static/Screenshot_2024-03-15_at_16.52.17.png)
            
        3. If the detection is invalid, delete the track by clicking on the trash
            
            ![Screenshot 2024-03-15 at 16.53.25.png](static/Screenshot_2024-03-15_at_16.53.25.png)
            
4. **Download Cleaned Annotations**
    
    ![Screenshot 2024-03-15 at 16.54.04.png](static/Screenshot_2024-03-15_at_16.54.04.png)
    
    1. Click on the 💾 Icon
    2. Then click Download > Viame CSV and download the file
5. **Extract MaxN from Cleaned Annotations**
    
    ![Screenshot 2024-03-15 at 17.10.52.png](static/Screenshot_2024-03-15_at_17.10.52.png)
    
    1. Open this [Collab Notebook](https://colab.research.google.com/drive/1oiJgt1TZnBoKLi3PCZBKtiH0NnRsb-0Z?authuser=0#scrollTo=qfJdcsy_D5i1)
    2. Upload the original `output.csv` file and the cleaned viame file you downloaded in step 4
    3. Edit cell two and insert the names of the files
    4. Run both cells
    5. Close and reopen the 📁 icon (left side)
    6. You will see a `max_n.csv` file, which is your final CSV