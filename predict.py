import sys
from genlink.retrieval_cli import main
if __name__=='__main__':
    sys.argv.insert(1,'predict')
    main()
