from discord.ext import commands
import discord

from core.classes import Cog_Extension
import core.function as function
from utils import engTyping_end_fix, engTyping_rearrange, IsZhInputs
import translator_viterbi

class Events(Cog_Extension):
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        message.content = engTyping_end_fix(engTyping_rearrange(message.content.lower()))
        if IsZhInputs(message.content):
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Get message: {message.content}')
            translated_msg = ''.join(translator_viterbi.decode_sentence(message.content)[1])
            await message.reply(translated_msg)
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Send message: {translated_msg}')


async def setup(bot):
    await bot.add_cog(Events(bot))