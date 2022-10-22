from stockfish import Stockfish

'''
@param: fen string
@returns: list of dict of top 3 best moves

Dict structure is as such
{'Move': 'd2d4', 'Centipawn': 41, 'Mate': None}

FEN tells you whose move it is: notice the 'w' stands for White's turn, 'b' stands for Black's turn
'''
def top_moves(fen):
    # init stockfish engine
    stockfish = Stockfish()
    # larger the hash value better, keep power of 2
    stockfish.update_engine_parameters({"Hash": 2048})

    # check if fen is valid
    if stockfish.is_fen_valid(fen):
        # sets the position of fen
        stockfish.set_fen_position(fen)
        # return list of the top 3 moves
        return stockfish.get_top_moves(3)

test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
print(top_moves(test_fen))

