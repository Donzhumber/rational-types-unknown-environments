$pdf_mode = 1;
$out_dir = 'out';
$aux_dir = 'out';
$bibtex_use = 2;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 %O %S';

use Cwd qw(abs_path);
my $docdir = abs_path('.');
my $bibdir = abs_path('../../Main/tex');
my $bstdir = abs_path('../esp');
$ENV{'BIBINPUTS'} = "$docdir:$bibdir:" . ($ENV{'BIBINPUTS'} // '');
$ENV{'BSTINPUTS'} = "$docdir:$bstdir:" . ($ENV{'BSTINPUTS'} // '');
