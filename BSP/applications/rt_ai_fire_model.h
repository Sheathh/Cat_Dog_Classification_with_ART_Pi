#ifndef __RT_AI_FIRE_MODEL_H
#define __RT_AI_FIRE_MODEL_H

/* model info ... */

// model name
#define RT_AI_FIRE_MODEL_NAME			"fire"

#define RT_AI_FIRE_WORK_BUFFER_BYTES		(23856)

#define AI_FIRE_DATA_WEIGHTS_SIZE		(648312)

#define RT_AI_FIRE_BUFFER_ALIGNMENT		(4)

#define RT_AI_FIRE_IN_NUM				(1)

#define RT_AI_FIRE_IN_SIZE_BYTES	{	\
	((64 * 64 * 3) * 1),	\
}
#define RT_AI_FIRE_IN_1_SIZE			(64 * 64 * 3)
#define RT_AI_FIRE_IN_1_SIZE_BYTES		((64 * 64 * 3) * 1)
#define RT_AI_FIRE_IN_TOTAL_SIZE_BYTES		((64 * 64 * 3) * 1)



#define RT_AI_FIRE_OUT_NUM				(1)

#define RT_AI_FIRE_OUT_SIZE_BYTES	{	\
	((1 * 1 * 2) * 1),	\
}
#define RT_AI_FIRE_OUT_1_SIZE			(1 * 1 * 2)
#define RT_AI_FIRE_OUT_1_SIZE_BYTES		((1 * 1 * 2) * 1)
#define RT_AI_FIRE_OUT_TOTAL_SIZE_BYTES		((1 * 1 * 2) * 1)




#define RT_AI_FIRE_TOTAL_BUFFER_SIZE		//unused

#endif	//end
