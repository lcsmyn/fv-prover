theory jain_2-1 imports "AutoCorres.AutoCorres" begin

external_file "jain_2-1.c"
install_C_file "jain_2-1.c"
autocorres "jain_2-1.c"

context jain_2-1 begin
thm main'_def

theorem main_safety:
  "\<turnstile> {\<lambda>s. True} main' {\<lambda>ret s. True}"
  
