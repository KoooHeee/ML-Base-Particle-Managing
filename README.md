# Identifying Particle Management Points in Etch Process Based on The Tree-Based Models
Using a tree-based machine learning algorithm, we predict the level of particle contamination coming from Etch equipment. Based on the feature importance of this machine learning, we want to identify the recipe Steps and Sensors that contribute the most to particle generation to find new equipment management points.

## Dataset
We chose Etch process equipment among various semiconductor production equipment and used ICP Etch chamber.
Due to security issues, disclosure of the entire dataset and detailed Fault Detection and Classification (FDC) items, sensors, and steps is limited. For this reason, our preprocessing process is not disclosed.

Sample datasets are included in the 10 sample datasets. Etch chambers do not exist in a single chamber, but 4-6 chambers are dependent on one main equipment. 
- 'Equipment' refers to the Main Equipment.
- 'Carrier' is a kind of storage box that separates the wafer from the outside exposure, and the movement of the wafer is done through this carrier.
- 'Slot' refers to a number in the Carrier, numbered from 1 to 25. 
- 'Time' is the time the Wafer started the process in the Chamber.
- 'FDC' items are private due to security issues and consist of 920 items in total.
- 'Particles' is the number of particles measured by a separate inspection equipment after running the particle check recipe in the Etch chamber.

## Tree-based ML
We used Random forest, Extra trees classifier, and XGBoost as machine learning models to check feature importance.
