image_paths = {}
with open("tags.txt", 'r') as f:
    print("f : ", f)
    for line in f:
        print("line :", line)
        t = line.strip().split()
        image_path = t[0]
        print('t[0] : ' , t[0])
        #break
        parts = image_path.split('/')
        print("parts :", parts)
        if not parts[0] in image_paths:
            image_paths[parts[0]] = []
        image_paths[parts[0]].append(image_path)

print(image_paths)