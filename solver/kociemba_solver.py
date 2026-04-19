import kociemba

# Your cube (face order: U, D, F, B, L, R)
cube = [
    [['W','W','W'], ['W','W','W'], ['W','W','W']],  # U
    [['B','B','B'], ['B','B','B'], ['B','B','B']],  # R
    [['R','R','R'], ['R','R','R'], ['R','R','R']],  # F
    [['Y','Y','Y'], ['Y','Y','Y'], ['Y','Y','Y']],  # D
    [['G','G','G'], ['G','G','G'], ['G','G','G']],  # L
    [['O','O','O'], ['O','O','O'], ['O','O','O']]   # B
]

def get_color_to_face_map(cube):
    centers = {
        cube[0][1][1]: 'U',
        cube[1][1][1]: 'R',
        cube[2][1][1]: 'F',
        cube[3][1][1]: 'D',
        cube[4][1][1]: 'L',
        cube[5][1][1]: 'B',
    }
    return centers

def flatten_faces(cube):
    for face in cube:
        for row in face:
            for val in row:
                yield val


def cube_to_kociemba(cube):
    color_map = get_color_to_face_map(cube)

    state = "".join(color_map[c] for c in flatten_faces(cube))
    return state


def solve(cube):
    state = cube_to_kociemba(cube)
    print("Kociemba state:", state)
    return kociemba.solve(state)


print(solve(cube))