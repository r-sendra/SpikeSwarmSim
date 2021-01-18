# import numpy as np 


# def detect_collisions(self, object_dict):
#     collision_dict = {}
#     for name, obj in object_dict.items():
#         no_collision = True
#         if obj.moving:
#             new_pos = obj.pos + obj.delta_pos
#             for name2, obj2 in object_dict.items(): 
#                 if name != name2:
#                     if not (all(new_pos + self.radius < obj2.pos - .5 * np.array(obj2.w, obj2.h))\
#                     and all(new_pos + self.radius > obj2.pos + .5 * np.array(obj2.w, obj2.h))):
#                         no_collision = no_collision and False
#                     if all(new_pos + self.radius) < np.zeros(2)) and all(all(new_pos + self.radius) < 1000 * np.ones(2))):
#                         no_collision = no_collision and False
#         collision_dict[name] = no_collision
#     return collision_dict



    