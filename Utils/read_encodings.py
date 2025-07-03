import ast  

def _read_encodings(filename="/content/encodings.txt"):
    encodings = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line:  
                encodings.append(ast.literal_eval(line)) 
    return encodings