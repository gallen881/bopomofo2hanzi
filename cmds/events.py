from discord.ext import commands
import discord

from core.classes import Cog_Extension
import core.function as function
from utils import engTyping_end_fix, engTyping_rearrange, IsZhInputs, text_classifier
import translator_viterbi

class Events(Cog_Extension):
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        texts = []
        text = engTyping_end_fix(engTyping_rearrange(message.content.lower()))
        text_list = text_classifier(text)
        if IsZhInputs(text):
            texts = [text]
        elif text_list[0]:
            texts = text_list
        if texts:
            output = []
            for text in texts:
                function.print_detail('INFO', message.author, message.guild, message.channel, f'Get message: {text}')
                output.append(''.join(translator_viterbi.decode_sentence(text)[1]))
            translated_msg = ' '.join(output)
            await message.reply(translated_msg)
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Send message: {translated_msg}')


async def setup(bot):
    await bot.add_cog(Events(bot))