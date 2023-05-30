from discord.ext import commands
from core.classes import Cog_Extension

class Tools(Cog_Extension):
    @commands.command()
    async def ping(self, ctx: commands.Context):
        await ctx.send(f'ping: {self.bot.latency * 1000} (ms)')
        
async def setup(bot):
    await bot.add_cog(Tools(bot))