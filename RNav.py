import ast
from enum import Enum
from typing import List, NamedTuple, Callable, Optional
from math import sqrt
from SearchAlgorithems import depthFirstSearch, breadthFirstSearch, greedyBestFirst, astar, astar_bidirectional, limitedDepthFirstSearch, Node, node_to_path, limitedAstarSearch
import tkinter as tk

class Cell(str, Enum):
    Empty = " o "
    Blocked = " X "
    StartingCell = " S "
    GoalCell = " G "
    Path = " * "

class MapLocation(NamedTuple):
    x: int
    y: int


class Map:
    def __init__(self, rows, cols, Walls: List[MapLocation], StartLocation: MapLocation, Goals: List[MapLocation]):
        self.Rows = rows
        self.Cols = cols
        self.Walls = Walls
        self.Start: MapLocation = StartLocation
        self.Goals: List[MapLocation] = Goals
        #creates a grid of empty cells
        self.Grid: List[List[Cell]] = [[Cell.Empty for _ in range(cols)] for _ in range(rows)]
        #sets the walls
        try:
            for wall in Walls:
                self.Grid[wall.x][wall.y] = Cell.Blocked
        except IndexError as e:
            print(e)
            print("l",wall.x,wall.y)
        #set start
        self.Grid[self.Start.y][self.Start.x] = Cell.StartingCell# reversed x and y
        #set goals

        for each in self.Goals:
            self.Grid[each.y][each.x]=Cell.GoalCell

    #return a formated map to print
    def __str__(self) -> str:
        result = ""
        for row in self.Grid:
            result += "".join([cell.value for cell in row]) + "\n"
        return result

    def goal_test(self, ml: MapLocation) -> bool:
        return self.Grid[ml.y][ml.x] == Cell.GoalCell

    def successors(self, ml: MapLocation) -> List[MapLocation]:
        locations: List[MapLocation] = []
        # Define the order of movements
        movements_order = ["UP", "LEFT", "DOWN", "RIGHT"]

        for move in movements_order:

            if move == "UP" and ml.y - 1 >= 0 and self.Grid[ml.y - 1][ml.x] != Cell.Blocked:
                locations.append(MapLocation(ml.x, ml.y - 1))
            elif move == "LEFT" and ml.x - 1 >= 0 and self.Grid[ml.y][ml.x - 1] != Cell.Blocked:
                locations.append(MapLocation(ml.x - 1, ml.y))
            elif move == "DOWN" and ml.y + 1 < self.Rows and self.Grid[ml.y + 1][ml.x] != Cell.Blocked:
                locations.append(MapLocation(ml.x, ml.y + 1))
            elif move == "RIGHT" and ml.x + 1 < self.Cols and self.Grid[ml.y][ml.x + 1] != Cell.Blocked:
                locations.append(MapLocation(ml.x + 1, ml.y))
        return locations
    
    def mark(self, path: List[MapLocation]):
        for maze_location in path:
            self.Grid[maze_location.y][maze_location.x] = Cell.Path
        self.Grid[self.Start.y][self.Start.x] = Cell.StartingCell
        for goal in self.Goals:
            self.Grid[goal.y][goal.x] = Cell.GoalCell

    def clear(self, path: List[MapLocation]):
        for maze_location in path:
            self.Grid[maze_location.y][maze_location.x] = Cell.Empty
        self.Grid[Start.y][Start.x] = Cell.StartingCell
        self.Grid[Goals[0].y][Goals[0].x] = Cell.GoalCell


def manhattan_distance(goal: MapLocation) -> Callable[[MapLocation], float]:
    def distance(ml: MapLocation) -> float:
        xdist: int = abs(ml.x - goal.x)
        ydist: int = abs(ml.y - goal.y)
        return xdist + ydist
    return distance


    

def create_gui(root, m, f_name):
    def button_handler(algorithm):
        def inner():
            nonlocal m, f_name
            solution = None
            if algorithm == "DFS":
                solution = depthFirstSearch(m.Start, m.goal_test, m.successors)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            elif algorithm == "BFS":
                solution = breadthFirstSearch(m.Start, m.goal_test, m.successors)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            elif algorithm == "Greedy":
                h = manhattan_distance(m.Goals[0])
                solution = greedyBestFirst(m.Start, m.goal_test, m.successors, h)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            elif algorithm == "A*":
                h = manhattan_distance(m.Goals[0])
                solution = astar(m.Start, m.goal_test, m.successors, h)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            elif algorithm == "Bidirectional A*":
                h = manhattan_distance(m.Goals[0])
                solution = astar_bidirectional(m.Start, m.Goals[0], m.goal_test, m.successors, h)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            elif algorithm == "Limited DFS":
                solution = limitedDepthFirstSearch(m.Start, m.goal_test, m.successors, 5)
                printTerminal(solution, m, f_name, algorithm)
                root.destroy()
            printResult(solution, m, f_name, algorithm, canvas)

        return inner

    def printResult(solution: Optional[Node[MapLocation]], m: Map, name: str, algorithmName: str, canvas: tk.Canvas) -> None:
        if solution is None:
            print("No goal is reachable")
        else:
            path: List[MapLocation] = node_to_path(solution)
            m.mark(path)

            # Clear previous path on canvas
            canvas.delete("path")
            cell_size = 40  # Adjust based on your preference
            for idx, loc in enumerate(path):
                x0, y0 = loc.x * cell_size, loc.y * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                canvas.after(600)  # Delay in milliseconds for animation
                canvas.create_rectangle(x0, y0, x1, y1, fill='blue', tags="path")
                canvas.update()
                canvas.after(1000)  # Delay in milliseconds for animation
                if idx < len(path) - 1:
                    canvas.create_rectangle(x0, y0, x1, y1, fill='yellow', tags="path")


    cell_size = 40  # Adjust this based on your preference
    canvas_width = m.Cols * cell_size
    canvas_height = m.Rows * cell_size

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')  # Set canvas background color
    canvas.pack(pady=10, expand=True, fill=tk.BOTH)  # Add padding around the canvas, fill the parent widget

    # Draw the maze on the canvas
    for y, row in enumerate(m.Grid):
        for x, cell in enumerate(row):
            x0, y0 = x * cell_size, y * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if cell == Cell.Blocked:
                canvas.create_rectangle(x0, y0, x1, y1, fill='black')
            elif cell == Cell.StartingCell:
                canvas.create_rectangle(x0, y0, x1, y1, fill='green')
            elif cell == Cell.GoalCell:
                canvas.create_rectangle(x0, y0, x1, y1, fill='red')
            elif cell == Cell.Path:
                canvas.create_rectangle(x0, y0, x1, y1, fill='yellow')
    

    algorithms = [
        "DFS", "BFS", "Greedy", "A*", "Bidirectional A*", "Limited DFS"
    ]

    for alg in algorithms:
        button = tk.Button(root, text=alg, command=button_handler(alg), width=12)  # Adjust button width
        button.pack(pady=5, fill=tk.BOTH)  # Add padding between buttons, fill the parent widget

    # Add a quit button
    quit_button = tk.Button(root, text="Quit", command=root.destroy, width=12)  # Adjust button width
    quit_button.pack(pady=10, fill=tk.BOTH)  # Add padding after buttons, fill the parent widget

    # Center the window on the screen
    window_width = canvas_width # Add space for padding and buttons
    window_height = canvas_height + 280  # Adjust height based on canvas size and buttons
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width // 2) - (window_width // 2)
    y_position = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
    root.title("Robot Navigation")

    root.mainloop()
    return canvas


def printTerminal(solution: Optional[Node[MapLocation]], m: Map, name: str, algorithmName: str) -> None:
    if solution is None:
        print("No goal is reachable")
    else:
        path: List[MapLocation] = node_to_path(solution)
        m.mark(path)

        movements = []
        for i in range(len(path) - 1):
            curr_loc = path[i]
            next_loc = path[i + 1]
            if next_loc.x > curr_loc.x:
                movements.append("Right")
            elif next_loc.x < curr_loc.x:
                movements.append("Left")
            elif next_loc.y > curr_loc.y:
                movements.append("Down")
            elif next_loc.y < curr_loc.y:
                movements.append("Up")

        print(m)
        print(name, algorithmName)
        print("Goal:")
        for goal in m.Goals:
            print(goal)

        print("Number of movements:", len(movements))
        print("\nMovements:", movements, "\n")
        print("Do you want to visualize in a GUI window? (y/n)")
        choice = input("Enter your choice: ")
        if choice.lower() == "y":
            root = tk.Tk()
            root.title("Search Algorithms")
            create_gui(root, m, f.name)
            root.mainloop()
        m.clear(path)



def printTerminal1(solution: Optional[Node[MapLocation]], m: Map, name: str, algorithmName: str) -> None:
    if solution is None:
        print("No goal is reachable")
    else:

        path: List[MapLocation] = solution
        m.mark(path)

        movements = []
        for i in range(len(path) - 1):
            curr_loc = path[i]
            next_loc = path[i + 1]
            if next_loc.x > curr_loc.x:
                movements.append("Right")
            elif next_loc.x < curr_loc.x:
                movements.append("Left")
            elif next_loc.y > curr_loc.y:
                movements.append("Down")
            elif next_loc.y < curr_loc.y:
                movements.append("Up")

        print(m)
        print(name, algorithmName)
        print("Goal:")
        for goal in m.Goals:
            print(goal)

        print("Number of movements:", len(movements))
        print("\nMovements:", movements)
        animate_choice = input("Do you want to display animation? (Y/N)").lower()
        if animate_choice == 'y':
            root = tk.Tk()
            root.title("Search Algorithms")
            create_gui(root, m, f.name)
            root.mainloop() 
        m.clear(path)

if __name__ == "__main__":
    while True:
        print("\nRobot Navigation using Search Algorithms\n")
        print("Press 'Q' to quit or enter the file name to load a test case\n")
        choice = input("Enter your choice: ")
        if choice == 'q':
            break

        try:
            with open(choice, 'r') as f:
                content = f.readlines()
        except FileNotFoundError:
            print("Invalid file name!")
            continue
        Goals : List[MapLocation] = []
        Walls : List[MapLocation] = []
        Start: MapLocation = MapLocation(0, 0)  # Default value
        read_line =0
        for line in content:
            if content[0]and read_line==0:
                MapSize = line.strip()
                MapSize = ast.literal_eval(MapSize)
                n :int = int(MapSize[0])
                m :int= int(MapSize[1])
                read_line+=1
                continue

            if content[1] and read_line==1:
                StartPosCoordinates = line.strip()
                StartPosCoordinates = ast.literal_eval(StartPosCoordinates)
                Start : MapLocation = MapLocation(int(StartPosCoordinates[0]), int(StartPosCoordinates[1]))
                read_line+=1
                continue

            if content[2] and read_line==2:
                GoalPosCoordinates = line.strip()
                GoalPosCoordinates_list = GoalPosCoordinates.split("|")
                for i in range(len(GoalPosCoordinates_list)):
                    GoalPosCoordinates_list2 = GoalPosCoordinates_list[i].strip()
                    GoalPosCoordinates_list2 = ast.literal_eval(GoalPosCoordinates_list2)
                    Goals.append(MapLocation(int(GoalPosCoordinates_list2[0]),int(GoalPosCoordinates_list2[1])))
                read_line+=1
                continue

            if read_line>2:
                WallPos = line.strip()
                WallPos = ast.literal_eval(WallPos)
                Walls.append(MapLocation(WallPos[1],WallPos[0]))
                width = WallPos[2]
                height = WallPos[3]
                for i in range(0,height):
                    for j in range(0,width):
                        Walls.append(MapLocation(int(WallPos[1]+i),int(WallPos[0]+j)))
        m: Map = Map(rows=n, cols=m, Walls=Walls, StartLocation=Start, Goals=Goals)
        print("\nMap:")
        print(m)
        #the bottom code is for terminal use when GUI was not developed. Now it's Obsolete
        while True:
            print("Select your search algorithm:\n 1. Depth-First Search\n 2. Breadth-First Search\n 3. Greedy Breadth-First Search\n 4. A* Search\n 5. Bidirectional A* Search\n 6. Limited Depth First Search\n 7. Quit\n")
            choice = input("Enter your choice: ")

            if choice == "1":
                # DFS
                solution1: Optional[Node[MapLocation]] = depthFirstSearch(m.Start, m.goal_test, m.successors)
                printTerminal(solution1, m, f.name, "Depth-First Search")

            elif choice == "2":
                # BFS
                solution2: Optional[Node[MapLocation]] = breadthFirstSearch(m.Start, m.goal_test, m.successors)
                printTerminal(solution2, m, f.name, "Breadth-First Search")

            elif choice == "3":
                # Greedy
                h: Callable[[MapLocation], float] = manhattan_distance(m.Goals[0])
                solution3: Optional[Node[MapLocation]] = greedyBestFirst(m.Start, m.goal_test, m.successors, h)
                printTerminal(solution3, m, f.name, "Greedy Breadth-First Search")

            elif choice == "4":
                # A*
                h: Callable[[MapLocation], float] = manhattan_distance(m.Goals[0])
                solution4: Optional[Node[MapLocation]] = astar(m.Start, m.goal_test, m.successors, h)
                printTerminal(solution4, m, f.name, "A* Search")

            elif choice == "5":
                h: Callable[[MapLocation], float] = manhattan_distance(m.Goals[0])
                solution5: Optional[Node[MapLocation]] = astar_bidirectional(m.Start, m.Goals[0], m.goal_test, m.successors, h)
                printTerminal1(solution5, m, f.name, "Bidirectional A* Search")

            elif choice == "6":
                solution6: Optional[Node[MapLocation]] = limitedDepthFirstSearch(m.Start, m.goal_test, m.successors, 1)
                printTerminal(solution6, m, f.name, "Limited Depth First Search")
            
            elif choice == "7":
                break

            else:
                print("\nInvalid choice!\n")
                continue
