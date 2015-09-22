#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)


#include "generic/LeakyReLU.c"
#include "THGenerateFloatTypes.h"

DLL_EXPORT int luaopen_libann(lua_State *L)
{
   nn_FloatLeakyReLU_init(L);
   nn_DoubleLeakyReLU_init(L);
   return 1;
}                                                                                                                                                                                                                                                                         
