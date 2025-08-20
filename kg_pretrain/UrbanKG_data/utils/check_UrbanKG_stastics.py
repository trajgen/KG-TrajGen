"""
borough, area, road, RC, junction, JC, poi, PC
"""

with open("../UrbanKG/porto_10904/entity2id_porto.txt") as f:
    Borough = []
    Area = []
    Road = []
    RC = []
    Junction = []
    JC = []
    POI = []
    PC = []
    for line in f.readlines():
        if 'Borough/' in line:
            Borough.append(line)
        if 'Area/' in line:
            Area.append(line)
        if 'Road/' in line:
            Road.append(line)
        if 'RC/' in line:
            RC.append(line)
        if 'Junction/' in line:
            Junction.append(line)
        if 'JC/' in line:
            JC.append(line)
        if 'POI/' in line:
            POI.append(line)
        if 'PC/' in line:
            PC.append(line)

print(len(Borough))
print(len(Area))
print(len(Road))
print(len(RC))
print(len(Junction))
print(len(JC))
print(len(POI))
print(len(PC))

"""

"""
Ent = []
Rel = []
Triplet = []
Train = []
Valid = []
Test = []

with open("../UrbanKG/porto_10904/entity2id_porto.txt") as f:
    for line in f.readlines():
        Ent.append(line)

print("Ent:", len(Ent))

with open("../UrbanKG/porto_10904/relation2id_porto.txt") as f:
    for line in f.readlines():
        Rel.append(line)
print("Rel:", len(Rel))

with open("../UrbanKG//porto_10904/triplets_porto.txt") as f:
    for line in f.readlines():
        Triplet.append(line)
print("Triplet:", len(Triplet))

with open("../UrbanKG/porto_10904/train_porto.txt") as f:
    for line in f.readlines():
        Train.append(line)
print("Train:", len(Train))

with open("../UrbanKG/porto_10904/valid_porto.txt") as f:
    for line in f.readlines():
        Valid.append(line)
print("Valid:", len(Valid))

with open("../UrbanKG/porto_10904/test_porto.txt") as f:
    for line in f.readlines():
        Test.append(line)
print("Test:", len(Test))

# Ent = []
# Rel = []
# Triplet = []
# Train = []
# Valid = []
# Test = []
# 
# with open("../UrbanKG/CHI/entity2id_CHI.txt") as f:
#     for line in f.readlines():
#         Ent.append(line)
# 
# print("Ent:", len(Ent))
# 
# with open("../UrbanKG/CHI/relation2id_CHI.txt") as f:
#     for line in f.readlines():
#         Rel.append(line)
# print("Rel:", len(Rel))
# 
# with open("../../UrbanKG_Embedding_Model/data/CHI/triplets_CHI.txt") as f:
#     for line in f.readlines():
#         Triplet.append(line)
# print("Triplet:", len(Triplet))
# 
# with open("../../UrbanKG_Embedding_Model/data/CHI/train_CHI.txt") as f:
#     for line in f.readlines():
#         Train.append(line)
# print("Train:", len(Train))
# 
# with open("../../UrbanKG_Embedding_Model/data/porto/valid_CHI.txt") as f:
#     for line in f.readlines():
#         Valid.append(line)
# print("Valid:", len(Valid))
# 
# with open("../../UrbanKG_Embedding_Model/data/CHI/test_CHI.txt") as f:
#     for line in f.readlines():
#         Test.append(line)
# print("Test:", len(Test))


