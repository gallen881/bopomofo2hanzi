from discord.ext import commands
import discord

from core.classes import Cog_Extension
import core.function as function
from utils import engTyping_end_fix, engTyping_rearrange, IsZhInputs, text_classifier, engTyping_insert_split_char, split_char
import translator_viterbi

class Events(Cog_Extension):
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if message.author == self.bot.user or message.content.startswith('-') or message.author.bot:
            return
        texts = []
        text = engTyping_end_fix(message.content.lower())
        text_list = text_classifier(text) if len(text) >= 4 else ['']
        if IsZhInputs(text):
            texts = [engTyping_rearrange(text)]
        elif len(max([engTyping_insert_split_char(part).split(split_char) for part in text_list], key=len)) >= 2:
            texts = text_list
        if texts:
            output = []
            for text in texts:
                function.print_detail('INFO', message.author, message.guild, message.channel, f'Get message: {text}')
                r = translator_viterbi.decode_sentence(text)
                if r[0] > 0: output.append(''.join(r[1]))
            translated_msg = ' '.join(output)
            await message.reply(translated_msg)
            function.print_detail('INFO', message.author, message.guild, message.channel, f'Send message: {translated_msg}')


async def setup(bot):
    await bot.add_cog(Events(bot))