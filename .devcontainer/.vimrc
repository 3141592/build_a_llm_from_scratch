set number

if exists('$TMUX')
  let &t_SI = "\ePtmux;\e\e[6 q\e\\"
  let &t_EI = "\ePtmux;\e\e[2 q\e\\"
else
  let &t_SI = "\e[6 q"
  let &t_EI = "\e[2 q"
endif

set tabstop=4
set shiftwidth=4
set expandtab
set autoindent
set filetype=python
filetype plugin indent on

" Remember history and cursor position
set viminfo='100,<50,s10,h

augroup remember_cursor
  autocmd!
  autocmd BufReadPost *
    \ if line("'\"") > 0 && line("'\"") <= line("$") |
    \   execute "normal! g`\"" |
    \ endif
augroup END

" Format Python file with Black
"nnoremap <leader>f :w<CR>:!black %<CR>:e<CR>