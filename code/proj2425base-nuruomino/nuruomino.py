# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 58:
# 109260 Ana Santos
# 109264 Francisco Mendonça

from sys import stdin
import numpy as np
from search import (
    Problem,
    Node,
    depth_first_tree_search,
    breadth_first_tree_search,
    depth_first_graph_search,
    depth_limited_search,
    iterative_deepening_search,
    bidirectional_search,
    greedy_search
)

"""Peças e todas as suas posições/simetrias possíveis"""
pieces = {
    "L":[[[1,0], [1,2],[1,1]],
         [[0,1],[2,1],[1,1]], 
         [[1,1,1], [1,2,0]],
         [[1,1,1], [0,2,1]],
         [[1,1],[1,2],[1,0]],
         [[1,1], [2,1], [0,1]],
         [[1,2,0], [1,1,1]],
         [[0,2,1], [1,1,1]]
        ],

    "I":[[[1], [1], [1], [1]],
           [[1,1,1,1]]
        ],

    "T":[[[1,2], [1,1], [1,2]],
        [[2,1],[1,1],[2,1]],
        [[1,1,1], [2,1,2]],
        [[2,1,2], [1,1,1]],
        ],
    
    "S":[[[2,1,1],[1,1,2]],
        [[1,1,2], [2,1,1]],
        [[1,2], [1,1], [2,1]],
        [[2,1], [1,1], [1,2]]
        ]    
}

class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id 
        NuruominoState.state_id += 1

    def __lt__(self, other): 
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, grid, regions, x, actions, free_regions):
        self.grid = grid
        self.size = len(grid)
        self.regions = regions
        self.n_regions = len(regions)
        self.free_regions = free_regions
        self.x = x
        self.actions = actions
    
    def deep_copy_board(self):
        """Faz uma cópia do board"""
        new_grid = []
        for row in self.grid:
            new_row = []
            for cell in row:
                new_cell = {"valor": cell["valor"], "região": cell["região"]}
                new_row.append(new_cell)
            new_grid.append(new_row)
        new_regions = {reg: coords[:] for reg, coords in self.regions.items()}
        new_x = [item for item in self.x]
        new_actions = [list(item) for item in self.actions]
        new_free_regions = [item for item in self.free_regions]
        return Board(new_grid, new_regions, new_x, new_actions,new_free_regions)
        
    def valid(self, row:int, col:int) -> bool:
        """Verifica se o valor está dentro da grelha"""
        grid_size = self.size
        if 0 <= row < grid_size and 0 <= col < grid_size:
            return True
        return False
    
    def get_value(self, row:int, col:int) -> str:
        """Devolve o valor na coordenada pedida"""
        if self.valid(row,col):
            return self.grid[row][col]["valor"]

    def get_region(self, row:int, col:int) -> int:
        """Devolve a região na coordenada pedida"""
        if self.valid(row,col):
            return self.grid[row][col]["região"]

    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        add = [(-1,-1), (-1,0), (-1,1),
                (0,-1), (0,1),
                (1,-1), (1,0), (1,1)]
        adjacent_positions = []
        for r, c in add: 
            row_adj,col_adj = r+row, c+col
            if self.valid(row_adj, col_adj):
                adjacent_positions.append((row_adj,col_adj)) 
        return adjacent_positions
    
    def cross_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, excluindo diagonais."""
        add = [(-1,0), (0,-1), (0,1), (1,0)]
        cross_positions = []
        for r, c in add: 
            row_adj,col_adj = r+row, c+col
            if self.valid(row_adj, col_adj): 
                cross_positions.append((row_adj,col_adj)) 
        return cross_positions
    
    def occupied(self, row: int, col: int) -> bool:
        """Verifica se a célula está ocupada por alguma peça"""
        return self.get_value(row, col) in ('L', 'I', 'T', 'S')
    
    def corner_square_occupied(self, row:int, col:int) -> bool:
        """Verifica a formação de quadrados à volta da célula num canto"""
        adj_positions = self.adjacent_positions(row,col)
        if (self.occupied(*adj_positions[0]) and self.occupied(*adj_positions[1]) and self.occupied(*adj_positions[2])):
            return True
        return False
    
    def side_square_occupied(self, row:int, col:int) -> bool:
        """Verifica a formação de quadrados à volta da célula numa borda lateral"""
        adj_positions = self.adjacent_positions(row,col)
        squares = [
            (0,1,2),
            (2,3,4)
        ]
        for i1,i2,i3 in squares:
            if (self.occupied(*adj_positions[i1]) and self.occupied(*adj_positions[i2]) and self.occupied(*adj_positions[i3])):
                return True
        return False 
    
    def bottom_square_occupied(self, row:int, col:int) -> bool:
        """Verifica a formação de quadrados à volta da célula em baixo"""
        adj_positions = self.adjacent_positions(row,col)
        squares = [
            (0,1,3),
            (1,2,4)
        ]
        for i1,i2,i3 in squares:
            if (self.occupied(*adj_positions[i1]) and self.occupied(*adj_positions[i2]) and self.occupied(*adj_positions[i3])):
                return True
        return False
    
    def top_square_occupied(self, row:int, col:int) -> bool:
        """Verifica a formação de quadrados à volta da célula em cima"""
        adj_positions = self.adjacent_positions(row,col)
        squares = [
            (0,2,3),
            (1,3,4)
        ]
        for i1,i2,i3 in squares:
            if (self.occupied(*adj_positions[i1]) and self.occupied(*adj_positions[i2]) and self.occupied(*adj_positions[i3])):
                return True
        return False
    
    def square_occupied(self, row:int, col: int) -> bool:
        """Verifica a formação de quadrados à volta da célula"""
        adj_positions = self.adjacent_positions(row,col)
        squares = [
            (0,1,3),
            (1,2,4),
            (3,5,6),
            (4,6,7)
        ]
        for i1,i2,i3 in squares:
            if len(adj_positions) > max(i1,i2,i3):
                if (self.occupied(*adj_positions[i1]) and self.occupied(*adj_positions[i2]) and self.occupied(*adj_positions[i3])):
                    return True
        return False
    
    def check_square_cell(self, row:int, col: int) -> bool:
        """Verifica se a célula faz parte de um quadrado"""
        square=False
        max = self.size-1
        if (row==0 or row==max) and (col==0 or col==max):
            square=self.corner_square_occupied(row, col)
        elif row==0 and col!=0 and col!=max:
            square=self.top_square_occupied(row, col)
        elif row==max and col!=0 and col!=max:
            square=self.bottom_square_occupied(row, col)
        elif row!=0 and row!=max and (col==0 or col==max):
            square=self.side_square_occupied(row, col)
        else:
            square=self.square_occupied(row, col)
        return square
    
    def four_cells_regions(self):
        """Devolve uma lista de todas as regiões que têm tamanho quatro"""
        regioes_tamanho_quatro = []
        for regioes in self.regions:
            if len(self.regions[regioes]) == 4:
                regioes_tamanho_quatro.append(regioes) 
            else:
                self.free_regions.append(regioes)
        return regioes_tamanho_quatro
    
    def empty_region(self, region:int):
        """Verifica se a região está vazia"""
        cells = self.regions[region]
        for x, y in cells:
            if self.occupied(x,y):
                return False
        return True
    
    def find_empty_regions(self):
        """Devolve uma lista de regiões vazias"""
        empty_regions = []
        for region in self.regions:
            if self.empty_region(region):
                empty_regions.append(region)
        return empty_regions
    
    def piece_fits(self, region:int, piece: list, row:int, col:int, pt: str):
        """Verifica se uma peça consegue encaixar dentro de uma região específica"""
        line_size = len(piece[0])   
        collum_size = len(piece)
        cells = []
        for i in range(collum_size):
            for j in range(line_size):
                if piece[i][j] == 1:
                    x = row + i
                    y = col + j 
                    if (not self.valid(x,y) or self.get_region(x,y)!=region):
                        return (False, [])
                    if (self.occupied(x,y)):
                        return (False, [])
                    if (self.specific_exists_adjacent_pieces(x,y,pt)):
                        return (False, [])
                    if self.check_square_cell(x, y):
                        return (False, [])
                    if (x,y) in self.x:
                        return (False, [])
                    cells.append((x,y))
        return (True, cells)
    
    def check_squares_with_piece(self, cells: list, pt: str, region: int):
        """Verifica se ao colocar uma peça se forma algum quadrado à sua volta"""
        flag = True
        for (x,y) in cells:
            self.grid[x][y]["valor"]=pt
        for (x,y) in cells:
            if self.check_square_cell(x, y):
                flag = False
        for (x,y) in cells:
            self.grid[x][y]["valor"]=region
        return flag
    
    def put_piece(self, pt: str, piece: list, row:int, col: int):
        """Coloca uma peça"""
        grid = self.grid
        x_list = self.x
        line_size = len(piece[0])   
        collum_size = len(piece)
        for i in range(collum_size):
            for j in range(line_size):
                if piece[i][j] == 1:
                    x = row + i
                    y = col + j 
                    grid[x][y]["valor"] = pt
                if piece[i][j] == 2:
                    x = row + i
                    y = col + j
                    x_list.append((x,y))
        
    def try_four_piece(self, pt: str, region: int):
        """Tenta encaixar uma peça numa região com quatro células"""
        positions = pieces[pt] 
        region_cells = self.regions[region] 
        for position in positions: 
            collum_size = len(position[0])
            line_size = len(position)
            x, y = region_cells[0] 
            for i in range(line_size):
                for j in range(collum_size):
                    if position[i][j] == 1: 
                        topX = x - i 
                        topY = y - j
                        flag, cells = self.piece_fits(region, position, topX, topY, pt)
                        if flag and self.check_squares_with_piece(cells, pt, region):
                            self.put_piece(pt, position, topX, topY)
                            return True
        return False

    def try_actions(self, region: int):
        """Tenta encaixar uma peça"""
        actions = [] 
        region_cells = self.regions[region] 
        for tipo in pieces:
            positions = pieces[tipo] 
            for position in positions:
                collum_size = len(position[0])
                line_size = len(position)
                tested_positions = []
                for x, y in region_cells: 
                    for i in range(line_size):
                        for j in range(collum_size):
                            if position[i][j] == 1: 
                                topX = x - i 
                                topY = y - j
                                if (topX, topY) in tested_positions:
                                    continue
                                tested_positions.append((topX, topY))
                                flag, cells = self.piece_fits(region, position, topX, topY, tipo)
                                if flag and self.check_squares_with_piece(cells, tipo, region):
                                    actions.append([region,tipo,position,topX,topY])
        return actions

    def pieces_on_four_cells_regions(self):
        """Coloca peças nas regiões que têm quatro células"""
        four_cell_regions = self.four_cells_regions()
        for region in four_cell_regions:
            for tipo in pieces:               
                if self.try_four_piece(tipo,region):
                    break
        return None
    
    def specific_exists_adjacent_pieces(self, row:int, col: int, piece: str):
        """Verifica se existe alguma peça igual adjacente onde estamos a tentar colocar uma peça específica"""
        regiao = self.get_region(row,col)
        adjacent_positions = self.cross_positions(row,col)
        for x, y in adjacent_positions:
            if self.get_value(x,y) == piece and self.get_region(x,y) != regiao:
                return True
        return False
    
    def first_cell_filled(self):
        """Retorna a primeira célula preenchida"""
        size = self.size
        for x in range(0,size):
            for y in range(0,size):
                if self.occupied(x,y):
                    return (x,y)
        return 
    
    def all_regions_filled(self):
        """Verifica se todas as regiões estão preenchidas"""
        return len(self.free_regions) == 0
    
    def all_pieces_connected(self):
        """Verifica se todas as peças estão conectadas"""
        n_cells = 4 * self.n_regions  
        visited = set()
        start = self.first_cell_filled()
        if not start:
            return False 
        fila = [start]
        visited.add(start)
        while fila:
            x, y = fila.pop(0)
            for nx, ny in self.cross_positions(x, y):
                if (nx, ny) not in visited and self.occupied(nx, ny):
                    fila.append((nx, ny))
                    visited.add((nx, ny))
        return len(visited) == n_cells
    
    def get_connected_regions(self):
        """Retorna todas as regiões que estão conectadas"""
        size = self.size
        visited = set()
        groups = []
        for row in range(size):
            for col in range(size):
                if self.occupied(row,col) and (row,col) not in visited:
                    group = set()
                    fila = [(row,col)]
                    visited.add((row,col))
                    while fila:
                        crow, ccol = fila.pop(0)
                        group.add((crow,ccol))

                        for nrow, ncol in self.cross_positions(crow,ccol):
                            if (nrow,ncol) not in visited and self.occupied(nrow,ncol):
                                fila.append((nrow,ncol))
                                visited.add((nrow,ncol))
                    groups.append(list(group))
        return list(groups)
    
    def no_islands(self):
        """ Verifica se existem "ilhas" """
        groups = self.get_connected_regions()
        for group in groups:
            x, y = group[0]
            if not self.adjacent_to_free_regions(x, y):
                return False
        return True

    def first_cell(self, piece: str, row: int, col: int):
        """Encontra as coordenadas correspondentes ao início de uma peça no board"""
        line_size = len(piece[0])   
        collum_size = len(piece)
        for i in range(collum_size):
            for j in range(line_size):
                if piece[i][j] == 1:
                    x = row + i
                    y = col + j 
                    return (x,y)
        return
    
    def adjacent_to_free_regions(self, x: int, y: int):
        """Verifica se a peça colocada toca numa região livre"""
        n_cells = 4 * self.n_regions
        visited = set()
        start = (x,y)
        if not start:
            return False
        fila = [start]
        visited.add(start)
        while fila:
            x, y = fila.pop(0)
            for nx, ny in self.cross_positions(x, y):
                if (nx, ny) not in visited and self.occupied(nx, ny):
                    fila.append((nx, ny))
                    visited.add((nx, ny))
                elif self.get_region(nx, ny) in self.free_regions:
                    return True
        return False or n_cells == len(visited)
    
    
    @staticmethod
    def parse_instance():
        grid = []
        regions = {}
        x = []
        actions = []
        free_regions = []
        for line in stdin:
            row = line.strip().split()
            grid_row=[]
            for cell in row:
                grid_row.append({"valor": int(cell), "região": int(cell)})
            grid.append(grid_row)
        for i in range(len(grid)):
            for j in range(len(grid)):
                reg = grid[i][j]["região"]
                if reg not in regions:
                    regions[reg] = []
                regions[reg].append((i, j))
        regions_ordenadas = dict(sorted(regions.items(), key=lambda item: len(item[1])))
        board = Board(grid, regions_ordenadas, x, actions, free_regions)
        board.pieces_on_four_cells_regions()
        for reg in free_regions:
                action = board.try_actions(reg)
                for act in action:
                    actions.append(act)
        board.actions=actions
        return board
    
    def print_grid(self):
        grid = self.grid
        grid_size = self.size
        output = ""
        for i in range(0, grid_size):
            for j in range(0, grid_size):
                output+=str(grid[i][j]["valor"])
                if j<grid_size-1:
                    output+="\t"
            output+="\n"
        return output


class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = NuruominoState(board) 

    def sort_actions(self, actions: list):
        """Ordena as regiões por ordem das que têm menos ações"""
        frequency = {}
        for act in actions:
            region = act[0]
            if region in frequency:
                frequency[region] += 1
            else:
                frequency[region] = 1
        sorted_actions = sorted(actions, key=lambda act: frequency[act[0]])
        return sorted_actions
    
    def region_least_actions(self, actions: list):
        """Retorna a região com menos ações"""
        frequency = {}
        for act in actions:
            region = act[0]
            frequency[region] = frequency.get(region, 0) + 1
        min_freq = min(frequency.values())
        for region in frequency:
            if frequency[region]==min_freq:
                region_least_freq=region
                break 
        return [act for act in actions if act[0] == region_least_freq]
    
    def check_regions_wt_acts(self, state: NuruominoState, actions):
        """Vê se existem regiões sem ações"""
        regions_with_acts = set(act[0] for act in actions)
        for reg in state.board.free_regions:
            if reg not in regions_with_acts:
                return False
        return True

    def actions(self, state: NuruominoState):
        """Recebe como argumento um estado e retorna uma lista de ações 
        que podem ser executadas a partir desse estado"""
        """CORRESPONDE A PREENCHER UMA REGIÃO NO TABULEIRO COM A LETRA IDENTIFICADORA DE UMA PEÇA"""
        actions = []
        if state.board.actions!=[]:
            for act in state.board.actions:
                (region,tipo,position,topX,topY) = act
                if region not in state.board.free_regions:
                    continue
                flag, cells = state.board.piece_fits(region, position, topX, topY, tipo)
                if flag and state.board.check_squares_with_piece(cells, tipo, region):
                    actions.append(act)
            if not self.check_regions_wt_acts(state, actions):
                return []
        state.board.actions=actions
        if len(actions)!=0:
            actions=self.region_least_actions(actions)
        return actions 

    def result(self, state: NuruominoState, action):
        """Recebe como argumento um estado e uma ação e retorna o resultado 
        de aplicar essa ação a esse estado"""
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        (region, pt, piece, row, col) = action
        board = state.board
        new_board = board.deep_copy_board()
        new_board.put_piece(pt, piece, row, col)
        new_board.free_regions.remove(region)
        if not new_board.no_islands():
            new_board.actions=[]
        return NuruominoState(new_board)
        
    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        return state.board.all_regions_filled() and state.board.all_pieces_connected() 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

if __name__ == "__main__":
    board = Board.parse_instance()
    nuruomino = Nuruomino(board)
    goal = depth_first_tree_search(nuruomino)
    print(goal.state.board.print_grid())

  