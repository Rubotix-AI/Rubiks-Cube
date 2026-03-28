"""
processing :
    extract image from feed 
    denoise image
    detect vertical and horizontal lines
    detect contours
    return frame

filtering : 
    list of contours and their coordinates
    check for all of these conditions
        1. 4 sides to the polygon
        2. squarish shape
        3. area is greater than some threshold
        4. dominant color is one of the sticker colors
    if you have 9 candidates:
        if 8(centres) of them are within threshold distance from centre contour(centre):
            number each of them from top to bottom, left to right, starting from k to 8+k
            display a mini digitized version on top left corner of video feed
    return list of struct with cell rgb, cell coordinate, bounding box coords

main :
    create matrix mapping of cube (6 x 3 x 3) (note down the flattened index for each position)
    if c is pressed:
        get rgb, pos coords, bbox coords
        convert rgb to cielab and get rubiks cube color
        convert pos and bbox coords into relative positions in the cube
        convert each number assigned to cell earlier to its matrix index and then put it in
        move to next index batch starting from 9 to 17

render_cube : 
    if last index == 54 and after recording:
        quit opencv
        render interactive cube with data recorded from cam.

solver : 
    use kociemba to solve the cube and get the move list
    return move list and number of moves

arm : 
    take the list of moves
    convert each move into its IK commands
    concatenate all the commands together
    return final IK commands

environment : 
    set robot arm
    set table
    set rubiks cube in its recorded unscrambled state
    return env

simulation : 
    take env and IK commands
    start robosuite
    run IK commands in robosuite

"""