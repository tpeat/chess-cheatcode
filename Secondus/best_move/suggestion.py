from stockfish import Stockfish

# fetch fen from CV
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

stockfish = Stockfish(path='/opt/homebrew/bin/stockfish')
# larger the hash value better, keep power of 2
stockfish.update_engine_parameters({"Hash": 2048})

# check if fen is valid
if stockfish.is_fen_valid(fen):
    # sets the position of fen
    stockfish.set_fen_position(fen)
    # 
    print(stockfish.get_top_moves(3))
