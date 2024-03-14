

COCO_TO_NuScenes = {
    'class_num':6,
    'masks':
    [False,True,True,True,False,True,True,True,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,False,True,True,False,False,False,False,True,False,False,True,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,True,True,True,True,False,False,True,False,True,False ], 
    
    'classes':
    [6,0,0,0,6,0,0,0,6,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,4,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,4,5,5,6,4,4,6,6,6,6,2,6,6,1,6,6,6,6,6,6,4,4,4,4,4,4,4,4,4,5,4,6,6,6,6,6,2,5,3,3,6,6,4,6,4,6],

    'Mapping':
    {
            2:0, 3:0, 4:0, 6:0, 7:0, 8:0,
            101:1,
            98:2, 124:2,
            127:3, 126:3,
            10:4, 11:4, 12:4, 13:4, 14:4, 57:4, 88:4, 92:4, 93:4, 108:4, 109:4, 110:4, 111:4, 112:4, 113:4, 114:4, 115:4, 116:4, 118:4, 130:4, 132:4,
            89:5, 90:5, 117:5, 125:5, 
    },

    }


COCO_TO_A2D2_SKITTI = {
    'class_num':10,
    'masks':
    [
True,True,True,True,False,True,False,True,False,True,False,True,True,False,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,True,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,True,True,True,True,False,False,True,True,False,False
    ],

    'classes':
    [
3,2,0,2,10,0,10,1,10,9,10,9,9,10,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,8,8,10,7,9,10,10,10,10,10,10,10,4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,8,10,10,10,10,10,10,6,8,8,8,10,10,7,8,10,10
    ],

    'Mapping':
    {
        3:0, 6:0,
        8:1,
        2:2,
        1:3,
        101:4,
        #:5,  # None Exist
        124:6, 
        92:7, 130:7, 
        89:8, 90:8, 117:8, 125:8, 126:8, 127:8, 131:8,
        10:9, 12:9, 13:9, 15:9, 16:9, 17:9, 18:9, 19:9, 20:9, 93:9

    }
}

COCO_TO_VKITTI_SKITTI = {
    'class_num':6,
    'masks':
    [
False,False,True,False,False,False,False,True,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,True,False,True,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,True,True,False,False,True,False,False,False
    ],

    'classes':
    [
6,6,5,6,6,6,6,4,6,3,3,3,3,3,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0,0,6,1,3,6,6,6,6,6,6,6,2,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,0,6,6,6,6,6,6,6,6,0,0,6,6,1,6,6,6
    ],

    'Mapping':
    {
            89:0, 90:0, 117:0, 126:0, 127:0,
            92:1, 130:1,
            101:2, 
            10:3, 11:3, 12:3, 13:3, 14:3, 93:3, 
            8:4,
            3:5,
    }

} 

