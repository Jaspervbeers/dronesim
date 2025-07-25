(HDBeetle)
Model ID: 			MDL-HDBeetle-NN-II-NGP101
				 NN = Nominal configuration (no failure)
				 II = Indoor & OJF
				 NGP = (N)o normalization, with (G)ravity, (P)olynomial model
				 101 -> Linear w.r.t rotor speeds (Affine in the controls!)
Speed range:			0 - 5 [m/s]
Model created with: 		Python 3.8.12
				(may not work with other versions, especially versions outside 3.8)

IMPORTANT NOTE:
-When in hover (with only attitude and altitude control), the rotor speeds are not all at the same value as would be expected. 

-This is because the c.g. of the quadrotor is not at the (geometric) center of the quadrotor. The difference in rotor speeds was checked with a hovering flight and the hovering results are approximately the same (note that the exact values depend on the battery used and the placement thereof on the quadrotor). But indeed, rotor 1 is highest, followed by 4, 3 and then 2. 

-If there is velocity control enabled as well, then a difference in rotor speeds is also seen, although there is also an offset in attitude. The angle offset is due to the controller attempting to correct for a mismatch in c.g. location w.r.t where it expects it to be (i.e. @geometric center). 

-Note that such 'drifting' behaviour is also seen on the actual HDBeetle platform when trying to hover. To correct for this, users can try implement a position contorller to prevent small drifts. 

REMOVAL OF c.g. OFFSET
-As the c.g. of the platform shifts between flights (due to variability in detachable battery position) and even within a flight, it is difficult to determine a true c.g. location. Instead, an 'average' position is derived per model from its constituent training data.

-The effect of this c.g. offset, in the context of the forces and moment offsets, can be estimated using the model.checkOffsets() function of the droneModel class. This will create an FMOffsets.json file to store the estimated offsets. This file is then used on subseuqent simulation runs to set the offsets (instead of estimating them again). 

-If users instead prefer to keep the offsets, then they may toggle NoOffset=True when initializing the droneModel.Model object. 
